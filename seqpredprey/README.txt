README

This repository contains the code required to reproduce the experimental environment, training procedure, and evaluation pipeline described in the associated paper.

## Contents

* `seqpredprey_env.py` – Custom Gymnasium predator–prey environment.
* `train.py` – PPO training script used for all experiments.
* `test.py` – Evaluation script used to generate the datasets reported in the paper.

## Installation

After cloning the repository, install the environment from the repository root using

```bash
pip install -r requirements.txt

pip install -e .
```

## Training

The experiments reported in the paper were obtained by running
multiple independent training replications across combinations of:

* Reward modality:
  * Selfish (`coop=False`)
  * Collective (`coop=True`)

* Speed regime:
  * Same-speed (`var=False`)
  * Varied-speed (`var=True`)

* Perception regime:
  * Normal (`see_ally=True`)
  * Blind (`see_ally=False`)

* Random seed:
  * Multiple independent replications per condition.

train.py trains a single replication at a time. To reproduce
the full dataset reported in the paper, the script must be executed
repeatedly with different combinations of these parameters.

For example:

```bash
python train.py --coop True --var True --see_ally False --idx 1
```

trains one Blind Collective-Varied (CVB) replication.

Training produces PPO model checkpoints and files of the form:

`<condition>_preferred_speed.txt`

These files store the preferred prey speed used for evaluation, defined as the highest prey speed at which the success criterion was achieved during training.

## Evaluation

Run:

```bash
python test.py
```

The evaluation script expects to evaluate all trained replications for a given experiment. By default:

Normal conditions use seeds 1–30 (`SEEDS_NORMAL`).
Blind conditions use seeds 1–10 (`SEEDS_BLIND`).

Modify SEEDS_NORMAL and SEEDS_BLIND in test.py as needed.

The corresponding trained models must already exist in the expected directory structure.

The code variable `soft_flip` corresponds to the Flipped evaluation condition described in the manuscript.

The evaluation script generates the raw datasets used in the analyses reported in the paper, including:

- Episode-level performance records across prey-speed sweeps
- Episode-level performance records at preferred prey speed
- Aggregate catch-rate summaries across prey speeds
- Aggregate preferred-speed performance summaries
- Predator-specific catch attribution for each evaluation episode
- Predator sector-occupancy frequencies relative to the prey for each evaluation episode
