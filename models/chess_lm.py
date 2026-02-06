from typing import Optional
from abc import ABC

import torch
from lightning import LightningModule

from transformers import get_linear_schedule_with_warmup


class ChessLM(LightningModule, ABC):
    def __init__(self,
                 init_lr: float = 3e-4,
                 num_training_steps: Optional[int] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Subclasses should override
        self.model = None

        self.init_lr = init_lr
        self.num_training_steps = num_training_steps
        self.current_epoch_steps = 0

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                *args, **kwargs
    ) -> torch.Tensor:
        out = self.model(input_ids, return_dict=False)

        if isinstance(out, tuple):
            lm_logits = out[0]
        else:
            lm_logits = out

        if labels is not None:

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
            optimizer, num_warmup_steps=0.1 * self.num_training_steps,
            num_training_steps=self.num_training_steps)

        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]
