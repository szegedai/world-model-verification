import math
import os
import sys
from typing import Optional
from argparse import ArgumentParser

import chess
import torch
import numpy as np
from lightning import seed_everything
from torch import nn
from tqdm import tqdm

from data_utils.line_dataset import LineByLineTextDataset
from data_utils.data_collator import DataCollatorForLanguageModeling
from data_utils.chess_tokenizer import ChessTokenizer
from models import get_model, setup_probe
from utils import freeze
from utils.probing import convert_fen_to_labels, get_activations_with_fens
from train_utils import probe_loss_and_acc_stats


def _parse_args(args):
    parser = ArgumentParser("Board state probe training")

    # Model
    parser.add_argument("-m", "--model", type=str, default="chessgpt2_base", dest="model_type")
    parser.add_argument("-mp", "--model-path", type=str, required=True, dest="model_path")
    parser.add_argument("-l", "--layer", type=str, default="model.transformer.h.11", dest="layer_name")

    # Probe
    parser.add_argument("-pt", "--probe-type", type=str, default="linear", choices=["linear", "nonlinear"], dest="probe_type")

    # Training
    parser.add_argument("-e", "--epochs", type=int, default=3, dest="n_epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("-pb", "--probe-batch-size", type=int, default=4096, dest="probe_batch_size")
    parser.add_argument("-gpe", "--games-per-epoch", type=int, default=10000, dest="games_per_epoch")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-2, dest="learning_rate")
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-3, dest="weight_decay")
    parser.add_argument("-gpu", "--gpu-id", type=int, default=0, dest="gpu")
    parser.add_argument("-workers", "--num-workers", type=int, default=8, dest="n_workers")
    parser.add_argument("-sf", "--save-folder", type=str, default="./checkpoints/probes", dest="save_folder")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")
    parser.add_argument("-epv", "--epochs-per-val", type=int, default=1, dest="epochs_per_val")
    parser.add_argument("--mine-yours", action="store_true", default=False, dest="mine_yours")
    parser.add_argument("--mask-positions", type=str, default=None, choices=[None, "only_black", "only_white"], dest="mask_positions")
    parser.add_argument("--n-tokens-per-position", type=int, default=1, dest="n_tokens_per_position")
    parser.add_argument("--rap-prob", type=float, default=0.0, dest="rap_prob")
    parser.add_argument("--piece-loss-multiplier", type=float, default=1.0, dest="piece_loss_multiplier")

    # Dataset access
    parser.add_argument("--vocab-file", type=str, dest="vocab_file")
    parser.add_argument("--data-file", type=str, dest="data_file")

    return parser.parse_args(args)


def mask_positions(fens, mask_type: Optional[str] = None):
    if not mask_type:
        return [True for _ in fens]
    elif mask_type == "only_black":
        return [chess.Board(fen=fen).turn == chess.BLACK for fen in fens]
    elif mask_type == "only_white":
        return [chess.Board(fen=fen).turn == chess.WHITE for fen in fens]
    else:
        raise ValueError("mask_type can only be None, 'only_white' or "
                         f"'only_black', not {mask_type}")


def main(conf):
    seed_everything(conf.seed)


    # Model
    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"
    model = get_model(conf.model_type, checkpoint_path=conf.model_path)
    model.eval()
    freeze(model)
    model.to(device)
    layer_name = conf.layer_name

    # Dataset
    max_positions = 800
    tokenizer = ChessTokenizer(conf.vocab_file)
    dataset = LineByLineTextDataset(tokenizer,
                                    conf.data_file,
                                    max_positions,
                                    rap_prob=conf.rap_prob)

    collate_fn = DataCollatorForLanguageModeling(tokenizer)

    # Data subsets and dataloaders
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [.9, .1])
    train_loader = torch.utils.data.DataLoader(train_dataset, conf.batch_size, True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, conf.batch_size, False, collate_fn=collate_fn)

    # Probe
    n_tokens_per_position = conf.n_tokens_per_position
    linear = conf.probe_type == "linear"
    probe = setup_probe(model,
                        layer_name,
                        device,
                        linear=linear,
                        n_tokens_per_position=n_tokens_per_position)
    probe.to(device)
    mine_yours = conf.mine_yours

    # Train params
    probe_batch_size = conf.probe_batch_size
    optimizer = torch.optim.AdamW(probe.parameters(),
                                  lr=conf.learning_rate,
                                  weight_decay=conf.weight_decay,
                                  betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    os.makedirs(conf.save_folder, exist_ok=True)


    stored_activations = []
    stored_fens = []

    # Training loop
    for epoch in range(1, conf.n_epochs + 1):
        batch_remaining_games = conf.games_per_epoch

        # Train step
        probe.train()

        sum_corr = 0
        sum_pred = 0
        sum_corr_piece = 0
        sum_pred_piece = 0
        losses = []

        train_pbar = tqdm(train_loader,
                          desc="Loss: NaN, accuracy: 0.0, piece accuracy: 0.0",
                          total=math.ceil(conf.games_per_epoch / conf.batch_size))

        for data in train_pbar:
            if batch_remaining_games <= 0:
                break

            max_idx = min(batch_remaining_games, len(data["input_ids"]))
            data = {
                "input_ids": data["input_ids"][:max_idx],
                "separator_ind": data["separator_ind"][:max_idx]
            }

            optimizer.zero_grad()

            activations, fens = get_activations_with_fens(model, tokenizer, data, layer_name, device, n_tokens_per_position)

            stored_activations.extend(activations.detach().cpu())
            stored_fens.extend(fens)

            while (len(stored_activations) >= probe_batch_size) or (max_idx <= conf.batch_size and len(stored_activations) > 0):
                activation_batch = stored_activations[:probe_batch_size]
                fen_batch = stored_fens[:probe_batch_size]
                mask = mask_positions(fen_batch, conf.mask_positions)

                stored_activations = stored_activations[probe_batch_size:]
                stored_fens = stored_fens[probe_batch_size:]

                activation_batch = torch.stack(activation_batch).to(device)
                labels = [convert_fen_to_labels(fen, mine_yours=mine_yours)
                          for fen in fen_batch]
                labels = torch.stack(labels).to(device)

                activation_batch = activation_batch[mask]
                labels = labels[mask]

                probe_out = probe(activation_batch)

                (loss, n_corr, n_pred, n_corr_piece, n_pred_piece) = probe_loss_and_acc_stats(probe_out, labels, loss_fn, conf.piece_loss_multiplier)
                loss.backward()
                optimizer.step()
                scheduler.step()

                sum_corr += n_corr
                sum_pred += n_pred
                sum_corr_piece += n_corr_piece
                sum_pred_piece += n_pred_piece
                losses.append(loss.item())
                train_pbar.set_description(f"Loss: {loss}, accuracy: {n_corr / n_pred}, piece accuracy: {n_corr_piece / n_pred_piece}")

            batch_remaining_games -= len(data["input_ids"])

        print(f"Epoch #{epoch} train:")
        print(f"  Mean loss: {np.mean(np.array(losses))}")
        print(f"  Mean accuracy: {sum_corr / sum_pred}")
        print(f"  Mean piece accuracy: {sum_corr_piece / sum_pred_piece}")

        torch.save(probe.state_dict(), os.path.join(conf.save_folder, f"epoch_{epoch}.pt"))

        # Val step
        if epoch % conf.epochs_per_val != 0: continue

        probe.eval()

        sum_corr = 0
        sum_pred = 0
        sum_corr_piece = 0
        sum_pred_piece = 0
        losses = []

        for data in tqdm(val_loader):
            activations, fens = get_activations_with_fens(model, tokenizer, data, layer_name, device, n_tokens_per_position)
            mask = mask_positions(fens, conf.mask_positions)
            activations = activations.to(device)
            activations = activations[mask]
            probe_out = probe(activations)
            labels = torch.stack([convert_fen_to_labels(fen, mine_yours=mine_yours)
                                  for fen in fens]).to(device)
            labels = labels[mask]
            (loss, n_corr, n_pred, n_corr_piece, n_pred_piece) = probe_loss_and_acc_stats(probe_out, labels, loss_fn, conf.piece_loss_multiplier)

            sum_corr += n_corr
            sum_pred += n_pred
            sum_corr_piece += n_corr_piece
            sum_pred_piece += n_pred_piece
            losses.append(loss.item())

        print(f"Epoch #{epoch} val:")
        print(f"  Mean loss: {np.mean(np.array(losses))}")
        print(f"  Mean accuracy: {sum_corr / sum_pred}")
        print(f"  Mean piece accuracy: {sum_corr_piece / sum_pred_piece}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(_parse_args(args))
