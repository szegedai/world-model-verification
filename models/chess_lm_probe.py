from typing import Optional
from abc import ABC

import torch
from torch import nn
from lightning import LightningModule

from transformers import get_linear_schedule_with_warmup

from utils.probing import get_move_end_activations



def probe_loss(preds: torch.Tensor,
               labels: torch.Tensor,
               loss_fn: torch.nn.modules.loss._Loss,
               piece_loss_multiplier: float = 1.0
) -> torch.Tensor:
    # Hard predictions + labels
    preds = preds.view(-1, preds.size(-1))
    labels = labels.view(-1, labels.size(-1))
    argmax_labels = torch.argmax(labels, 1)

    # Set up weight vector
    weight = torch.ones(argmax_labels.shape).to(labels.device)
    weight[argmax_labels != 0] = piece_loss_multiplier

    loss = loss_fn(preds, labels)
    loss = (loss * (weight / weight.sum())).sum()

    return loss



class ChessLMProbe(LightningModule, ABC):
    def __init__(self,
                 init_lr: float = 3e-4,
                 num_training_steps: Optional[int] = None,
                 probe: nn.Module = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Subclasses should override
        self.model: nn.Module = None # type: ignore
        self.head: nn.Module = None # type: ignore

        self.init_lr = init_lr
        self.num_training_steps = num_training_steps
        self.current_epoch_steps = 0
        self.probe = probe

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                board_states: Optional[torch.Tensor] = None,
                separator_ind: Optional[torch.Tensor] = None,
                *args, **kwargs
    ) -> torch.Tensor:
        act = self.model(input_ids)[0]
        out = self.head(act)

        if separator_ind is not None:
            move_end_acts = []
            for a, s in zip(act, separator_ind):
                move_end_acts.extend(get_move_end_activations(a, s))
            move_end_acts = torch.stack(move_end_acts)
            # move_end_acts = get_move_end_activations(act, separator_ind)
            board_state_pred = self.probe(move_end_acts)

        if isinstance(out, tuple):
            lm_logits = out[0]
        else:
            lm_logits = out

        if (labels is not None and
            board_states is not None and
            separator_ind is not None):

            # `loss` here is the next-token loss
            if len(labels.shape) == 2:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                num_terms = torch.sum(shift_labels != -100)

                if num_terms:
                    loss = loss / num_terms
                else:
                    loss = 0.0

            else:
                # Labels are probability distributions instead of hard next-tokens
                shift_logits = lm_logits.contiguous()
                shift_labels = labels.contiguous()
                new_shape = (shift_logits.shape[0] * shift_logits.shape[1], shift_logits.shape[2])
                shift_labels = shift_labels.reshape(new_shape)
                shift_logits = shift_logits.reshape(new_shape)
                eps = 1e-12
                shift_logits = torch.log_softmax(shift_logits, -1)
                loss = (-torch.sum(shift_labels * shift_logits + eps, dim = 1)).mean()


            # Board state loss calculation
            probe_loss_fn = nn.CrossEntropyLoss(reduction="none")
            board_state_loss = probe_loss(board_state_pred,
                                          board_states,
                                          probe_loss_fn,
                                          piece_loss_multiplier=1)

            return loss + board_state_loss

        # Standard loss - only for perplexity evaluation!
        elif labels is not None:
            if len(labels.shape) == 2:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                num_terms = torch.sum(shift_labels != -100)

                if num_terms:
                    loss = loss / num_terms
                else:
                    loss = 0.0

            else:
                # Labels are probability distributions instead of hard next-tokens
                shift_logits = lm_logits.contiguous()
                shift_labels = labels.contiguous()
                new_shape = (shift_logits.shape[0] * shift_logits.shape[1], shift_logits.shape[2])
                shift_labels = shift_labels.reshape(new_shape)
                shift_logits = shift_logits.reshape(new_shape)
                eps = 1e-12
                shift_logits = torch.log_softmax(shift_logits, -1)
                loss = (-torch.sum(shift_labels * shift_logits + eps, dim = 1)).mean()
            return loss

        else:
            if separator_ind is not None:
                return lm_logits, board_state_pred
            else:
                return lm_logits

    def training_step(self, batch, batch_ids):
        loss = self(**batch)

        train_loss = loss
        train_log = {'loss/train_loss': loss}

        self.current_epoch_steps += batch["input_ids"].shape[0]
        return {'loss': train_loss, 'log': train_log}

    def validation_step(self, batch, batch_ids, split="val"):
        input_ids, labels = batch["input_ids"], batch["labels"]
        loss = self(**batch)
        # Removing labels for which losses are not calculated.
        batch_tokens = torch.sum(labels != -100)
        val_log = {f'loss/{split}_loss': loss.detach()}
        return {f'{split}_loss': loss.detach(), 'log': val_log, 'batch_tokens': batch_tokens,
                'batch_size': input_ids.shape[0]}

    def test_step(self, batch, batch_ids):
        return self.validation_step(batch, batch_ids, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.init_lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * self.num_training_steps, # type: ignore
            num_training_steps=self.num_training_steps)

        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]
