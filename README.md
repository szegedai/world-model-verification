# Verification of the Implicit World Model in a Generative Model via Adversarial Sequences (ICLR 2026)

This repository contains the code to reproduce the results presented in the ICLR 2026 paper 'Verification of the Implicit World Model in a Generative Model via Adversarial Sequences'.

Links to all resources used in our experiments, such as model checkpoints and datasets, can be found below.

## Resources

### Project Page
Coming soon...

### Checkpoints
Coming soon...

### Datasets
Coming soon...

### Demo
Coming soon...

### Citation

```
@inproceedings{
  balogh2026verification,
  title={Verification of the Implicit World Model in a Generative Model via Adversarial Sequences},
  author={Andr{\'a}s Balogh and M{\'a}rk Jelasity},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=BLOIB8CwBI}
}
```

## Scripts

### Preliminaries

To match the environment used in our experiments, create a Conda environment using the provided `environment.yml` file:
```sh
conda env create -f environment.yml
```

Note: some scripts have command line arguments that are not used in the paper. These are not mentioned here either, and should be left on their default value. Most of the command line arguments have shorthands, their longer versions are provided here.

### Model Training
`exec_train.py` should be used to train sequence models. Its command line arguments are the following:
- `--model`: Type of model to be trained. `chessgpt2_base` or `chessgpt2_base_probe` were used in the paper, the latter for the joint probe (+JP) objective.
- `--valid-token-probs-target`: Boolean switch to control the probability distribution (PD) objective. If specified, PD will be used, if not (default), the hard next-token prediction (NT) objective will be used.
- `--epochs`: Number of epochs.
- `--batch-size`: Batch size.
- `--real-batch-size`: Real batch size, used to control gradient accumulation. The number of batches over which gradient accumulation occurs is `real_batch_size / batch size`, this fraction should be a whole number.
- `--train-dataset-size`: Size of the training dataset.
- `--data-dir`: Directory of the dataset.
- `--vocab_dir`: Directory of the vocabulary.
- `--seed`: Seed for reproducibility. We used 0 in all our experiments.
- `--save-folder`: The model will be saved here.
- `--gpu-ids`: Cuda device IDs to be used during training.
- `--num-workers`: Number of workers.

### Probe Training
`exec_train_probe.py` should be used to train linear board state probes. Its command line arguments are the following:
- `--model`: Type of model to be used as a frozen backbone. `chessgpt2_base` was used in the paper.
- `--model-path`: Path to the model checkpoint.
- `--layer`: Name of the probed layer, as it shows up in a model's `state_dict`. For `chessgpt2_base`, we used `model.transformer.h.11` (default).
- `--epochs`: Number of epochs.
- `--probe-batch-size`: Batch size for training the probe.
- `--games-per-epoch`: Number of games per one epoch of probe training.
- `--learning-rate`: Initial learning rate.
- `--weight-decay`: L2 weight decay.
- `--gpu-id`: Cuda device ID to be used during training, -1 for CPU.
- `--save-folder`: The probe will be saved here.
- `--seed`: Seed for reproducibility. We used 0 in all our experiments.
- `--data-file`: Dataset file location.
- `--vocab_file`: Vocabulary file location.
- `--epochs-per-val`: Number of epochs between each validation step.

### Adversarial Evaluation
`exec_simulate_games.py` should be used for adversarial evaluation. Its command line arguments are the following:
- **Positional** `model_path`: Path to the model checkpoint
- **Positional** `model_type`: Type of model to be trained. `chessgpt2_base` or `chessgpt2_base_probe` were used in the paper.
- `--data-file`: Dataset file location.
- `--vocab_file`: Vocabulary file location.
- `--n-games`: Number of games (unique prefixes) for evaluation.
- `--white-move-type`: Adversary implementation. Options are:
  - `random_move`: RM
  - `lm_move`: Uncorrected SM move (see Appendix E)
  - `lm_move_corrected`: SMM
  - `lm_move_adversarial`: AD
  - `oracle_illegal_move`: IMO
  - `oracle_probe_error`: BSO
- `--n-train-moves`: Number of moves **per player** in the warmup sequence. For example, for a warmup sequence of 10 moves, the value of this argument should be 5.
- `--save-file`: Path to save the detailed results.
- `--gpu`: Cuda device ID to be used during training, -1 for CPU.
- `--probe-path`: Path to the probe file for the BSO adversary. For `chessgpt2_base_probe`, the value of this argument is irrelevant.
- `--probe-layer-name`: Name of the probed layer, as it shows up in a model's `state_dict`. For `chessgpt2_base`, we used `model.transformer.h.11` (default). For `chessgpt2_base_probe`, the value of this argument is irrelevant.

### Perplexity Evaluation
This part is only referenced in the Appendix. `exec_eval_lm.py` should be used for perplexity evaluation. Its command line arguments are the following:
- `--model`: Type of model to be used as a frozen backbone. `chessgpt2_base` was used in the paper.
- `--model-path`: Path to the model checkpoint.
- `--batch-size`: Batch size.
- `--gpu-id`: Cuda device ID to be used during evaluation, -1 for CPU.
- `--seed`: Seed for reproducibility. We used 0 in all our experiments.
- `--data-file`: Dataset file location.
- `--vocab_file`: Vocabulary file location.

### Probe Accuracy Evaluation
This part is only referenced in the Appendix. `exec_eval_lm.py` should be used for perplexity evaluation. Its command line arguments are the following:
- `--model`: Type of model to be evaluated. `chessgpt2_base` was used in the paper.
- `--model-path`: Path to the model checkpoint.
- `--layer`: Name of the probed layer, as it shows up in a model's `state_dict`. For `chessgpt2_base`, we used `model.transformer.h.11` (default). For `chessgpt2_base_probe`, the value of this argument is irrelevant.
- `--probe-path`: Path to the probe file. For `chessgpt2_base_probe`, the value of this argument is irrelevant.
- `--data-file`: Dataset file location.
- `--vocab_file`: Vocabulary file location.
- `--gpu-id`: Cuda device ID to be used during evaluation, -1 for CPU.
- `--save-folder`: The results will be saved here.
- `--seed`: Seed for reproducibility. We used 0 in all our experiments.

### Probe-Model-Ground Truth Overlap Evaluation
This part is only referenced in the Appendix. `exec_eval_board_state_overlap.py` should be used for perplexity evaluation. Its command line arguments are the following:
- `--model`: Type of model to be evaluated. `chessgpt2_base` was used in the paper.
- `--model-path`: Path to the model checkpoint.
- `--layer`: Name of the probed layer, as it shows up in a model's `state_dict`. For `chessgpt2_base`, we used `model.transformer.h.11` (default). For `chessgpt2_base_probe`, the value of this argument is irrelevant.
- `--epsilon`: Epsilon threshold for implicit state recovery.
- `--probe-path`: Path to the probe file. For `chessgpt2_base_probe`, the value of this argument is irrelevant.
- `--data-file`: Dataset file location.
- `--vocab_file`: Vocabulary file location.
- `--gpu-id`: Cuda device ID to be used during evaluation, -1 for CPU.
- `--save-folder`: The results will be saved here.
