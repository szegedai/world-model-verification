import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightning import seed_everything

from models import get_model
from data_utils.chess_tokenizer import ChessTokenizer
from data_utils.data_collator import DataCollatorForLanguageModeling
from data_utils.line_dataset import LineByLineTextDataset



def _parse_args(args):
    parser = ArgumentParser("Chess LLM perplexity eval")

    # Model
    parser.add_argument("-m", "--model", type=str, default="chessgpt2_base", dest="model_type")
    parser.add_argument("-mp", "--model-path", type=str, required=True, dest="model_path")

    # Hyperparams
    parser.add_argument("-b", "--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--rap-prob", type=float, default=0.0, dest="rap_prob")
    parser.add_argument("-gpu", "--gpu-id", type=int, default=0, dest="gpu")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")

    # Dataset access
    parser.add_argument("--vocab-file", type=str, dest="vocab_file")
    parser.add_argument("--data-file", type=str, dest="data_file")

    return parser.parse_args(args)

def main(conf):
    print(f"Perplexity eval for {conf.model_path}")
    seed_everything(conf.seed)

    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"

    tokenizer = ChessTokenizer(vocab_file=conf.vocab_file)
    dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                    file_path=conf.data_file,
                                    block_size=800,
                                    rap_prob=conf.rap_prob)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    batch_size=64
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=data_collator)


    model = get_model(conf.model_type, checkpoint_path=conf.model_path)
    model.eval()
    model.to(device)

    max_length = 1024
    stride = 512

    # Perplexity evaluation
    # Based on https://huggingface.co/docs/transformers/perplexity
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    max_length = 1024
    stride = 512

    for data in tqdm(data_loader):
        seq_len = data["input_ids"].shape[1]

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = data["input_ids"][:, begin_loc:end_loc].to(device)
            target_ids = data["labels"][:, begin_loc:end_loc].to(device)
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                # Loss is negative log-likelihood
                loss = model(input_ids, labels=target_ids)

            num_valid_tokens = (target_ids != -100).sum().item()
            nll_sum += loss * num_valid_tokens
            n_tokens += num_valid_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    print(f"Perplexity: {ppl.item()}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(_parse_args(args))
