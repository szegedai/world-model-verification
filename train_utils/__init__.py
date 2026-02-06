from typing import Tuple

import torch


def probe_loss_and_acc_stats(preds: torch.Tensor,
                             labels: torch.Tensor,
                             loss_fn: torch.nn.modules.loss._Loss,
                             piece_loss_multiplier: float = 1.0
) -> Tuple[torch.Tensor, int, int, int, int]:
    # Hard predictions + labels
    preds = preds.view(-1, preds.size(-1))
    labels = labels.view(-1, labels.size(-1))
    argmax_labels = torch.argmax(labels, 1)
    argmax_preds = torch.argmax(preds, 1)

    # Set up weight vector
    weight = torch.ones(argmax_labels.shape).to(labels.device)
    weight[argmax_labels != 0] = piece_loss_multiplier

    loss = loss_fn(preds, labels)
    loss = (loss * (weight / weight.sum())).sum()

    piece_labels = argmax_labels[argmax_labels != 0]
    piece_preds = argmax_preds[argmax_labels != 0]

    sum_corr = torch.sum(argmax_preds == argmax_labels).item()
    sum_piece_corr = torch.sum(piece_preds == piece_labels).item()

    return loss, sum_corr, len(labels), sum_piece_corr, len(piece_labels)
