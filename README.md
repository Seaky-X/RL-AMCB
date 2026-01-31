````md
#RL-AMCB: Reinforcement Learning-Driven Adaptive
Configuration for Medical Consortium Blockchains

This project implements a **BIoMT-driven dynamic blockchain configuration** system that supports:

- **Channel selection + VRF**
- **Reinforcement learning (RL) configures `m/n` for the selected channel**
- **State-driven `α, β, γ`**
- **TxPool without eviction (no dropping/expiration removal)**

It supports:

- **Training set:** Set-A + Outcomes-a  
- **Test set:** Set-B

## Directory Structure

```text
physionet2012/
  set-a/
    132539.txt
    ...
  set-b/
    140501.txt
    ...
  Outcomes-a.txt
````

## Run

### Train (using Set-A)

```bash
pip install -r requirements.txt
python -m rl.train_rl_ppo --mode train --set_a_dir <path>/set-a --outcomes_a <path>/Outcomes-a.txt
```

### Evaluate (using Set-B, rollout only, no updates)

```bash
python -m rl.train_rl_ppo --mode eval --set_b_dir <path>/set-b
```

## Outputs

* Training/Evaluation logs: `logs/train_log.csv` or `logs/eval_log.csv`
* Plotting: `python -m logs.plot_metrics` (reads the most recent CSV)

```
```
