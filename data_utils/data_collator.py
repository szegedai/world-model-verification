import torch
from torch.nn.utils.rnn import pad_sequence


def _pad_with_tensor(seqs: torch.Tensor,
                     pad_tensor: torch.Tensor
) -> torch.Tensor:
    B = len(seqs)
    K = seqs[0].shape[1]
    N_max = max(s.shape[0] for s in seqs)

    batch = pad_tensor.expand(B, N_max, K).clone().to(dtype=seqs[0].dtype)

    for i, seq in enumerate(seqs):
        batch[i, :seq.size(0)] = seq.clone()

    return batch


class DataCollatorForLanguageModeling:

    def __init__(self, tokenizer, rap_no_grad=True, model_type='transformer'):
        self.tokenizer = tokenizer
        self.rap_no_grad = rap_no_grad
        self.model_type = model_type

    def __call__(self, examples):
        batch = self._tensorize_batch([example['input_ids'] for example in examples])
        batch_sep = self._tensorize_batch([torch.tensor(example['separator_ind']) for example in examples])
        labels = batch.clone().detach()
        # Remove pad tokens and start tokens from loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.tokenizer.bos_token_id] = -100

        if self.rap_no_grad and 'piece_type_posns' in examples[0]:
            for idx, example in enumerate(examples):
                labels[idx, example['piece_type_posns']] = -100

        if 'valid_token_distributions' in examples[0].keys():
            # valid_tk_dists_batch = self._tensorize_batch([example['valid_token_distributions'] for example in examples])

            pad_tensor = [0] * len(self.tokenizer.vocab)
            pad_tensor[self.tokenizer.pad_token_id] = 1
            valid_tk_dists_batch = _pad_with_tensor([example['valid_token_distributions'] for example in examples],
                                                    torch.tensor(pad_tensor))

            output_dict = {"input_ids": batch, "labels": valid_tk_dists_batch, 'separator_ind': batch_sep}
        else:
            output_dict = {"input_ids": batch, "labels": labels, 'separator_ind': batch_sep}

        # Instead of padding board states, just concatenate them here so that
        # the format matches the one used in decoder training
        if 'board_states' in examples[0].keys():

            batch_board_states = [example['board_states']
                                  for example in examples]
            output_dict['board_states'] = torch.cat(batch_board_states, dim=0)

        return output_dict

    def _tensorize_batch(self, examples):
        padded_sequence = pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.model_type != 'reformer':
            return padded_sequence
        else:
            max_len = padded_sequence.shape[1]
            # increased_len = 350 - max_len
            # additional_padding = torch.Tensor(padded_sequence.shape[0], increased_len).fill_(
            #     self.tokenizer.pad_token_id)
            # return torch.cat([padded_sequence, additional_padding.long()], dim=1)
            if max_len % 50 == 0:
                return padded_sequence
            else:
                increased_len = (max_len//50 + 1) * 50 - max_len
                additional_padding = torch.Tensor(padded_sequence.shape[0], increased_len).fill_(
                    self.tokenizer.pad_token_id)
                return torch.cat([padded_sequence, additional_padding.long()], dim=1)
