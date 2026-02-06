import math
import sys
from argparse import ArgumentParser

from lightning import Trainer, seed_everything

from data_utils.chess_data_module import ChessLMDataModule
from models import get_model


def _parse_args(args):
    parser = ArgumentParser("Chess LLM training")
    # Training hyperparams
    parser.add_argument("-m", "--model", type=str, default="chessgpt2_base", dest="model_type")
    parser.add_argument("-e", "--epochs", type=int, default=10, dest="n_epochs")
    parser.add_argument("-rb", "--real-batch-size", type=int, default=128, dest="real_batch_size")
    parser.add_argument("-b", "--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("-gpu", "--gpu-ids", type=int, nargs="*", default=[0], dest="gpu")
    parser.add_argument("-workers", "--num-workers", type=int, default=8, dest="n_workers")
    parser.add_argument("-tds", "--train-dataset-size", type=int, default=500000, dest="train_dataset_size")
    parser.add_argument("-sf", "--save-folder", type=str, default="./checkpoints", dest="save_folder")
    parser.add_argument("-rap", "--rap-prob", type=float, default=0.0, dest="rap_prob")
    parser.add_argument("--rap-no-grad", action="store_true", default=False, dest="rap_no_grad")
    parser.add_argument("--valid-token-probs-target", action="store_true", default=False, dest="valid_token_probs_target")

    # Dataset access
    parser.add_argument("--vocab-dir", type=str, dest="vocab_dir")
    parser.add_argument("--data-dir", type=str, dest="data_dir")

    parser.add_argument("-s", "--seed", type=int, default=0, dest="random_seed")

    return parser.parse_args(args)


def train(conf) -> None:
    seed_everything(conf.random_seed)

    train_percent_check = 1
    accumulate_grad_batches = conf.real_batch_size // conf.batch_size

    train_percent_check = 1 if train_percent_check is None else train_percent_check
    one_epoch_games = int(conf.train_dataset_size * train_percent_check)
    one_epoch_batches = int(math.ceil(
        one_epoch_games / (conf.batch_size * accumulate_grad_batches)))

    print(f"One epoch batches: {one_epoch_batches}")

    num_training_steps = one_epoch_batches * conf.n_epochs
    print("Number of training steps: %d" % num_training_steps)


    data_module = ChessLMDataModule(conf.data_dir,
                                    conf.vocab_dir,
                                    batch_size=conf.batch_size,
                                    num_workers=conf.n_workers,
                                    train_size=conf.train_dataset_size,
                                    rap_prob=conf.rap_prob,
                                    rap_no_grad=conf.rap_no_grad,
                                    valid_token_distribution_as_target=conf.valid_token_probs_target,
                                    board_state_auxiliary_target="probe" in conf.model_type)

    trainer = Trainer(accelerator="gpu",
                      devices=conf.gpu,
                      max_epochs=conf.n_epochs,
                      accumulate_grad_batches=accumulate_grad_batches,
                      default_root_dir=conf.save_folder)

    # model = ChessGPT2(num_training_steps=num_training_steps)
    model = get_model(conf.model_type, num_training_steps=num_training_steps)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    args = sys.argv[1:]
    train(_parse_args(args))
