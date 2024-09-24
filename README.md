# SAE Spelling

Code for the paper [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](https://arxiv.org/abs/2409.14507).

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install dependencies, run:

```
poetry install
```

## Project structure

This project is set up so that code which could be reused in other projects is in the main `sae_spelling` package, and code specific to the experiments in the paper are in `sae_spelling.experiments`. In the future, we may move some of these utilities to their own library. The `sae_spelling` package is structured as follows:

- `sae_spelling.feature_attribution`: Code for running SAE feature attribution experiments. Attribution tries to estimate the effect a latent has on the model output. Main exports include:
  - `calculate_feature_attribution()`
  - `calculate_integrated_gradient_attribution_patching()`
- `sae_spelling.feature_ablation`: Code for running SAE feature ablation experiments. This involves ablating each firing SAE latent on a prompt to see how it affects a downstream metric (e.g. if the model knows the first letter of a token). The main function in this module is:
  - `calculate_individual_feature_ablations()`
- `sae_spelling.probing`: Code for training logistic-regression probes in Torch. Some helpful exports from this module are:
  - `train_multi_probe()`: train a multi-class binary probe
  - `train_binary_probe()`: train a binary probe (same as the multi-class probe, but with only one class)
- `sae_spelling.prompting`: Code for generating ICL prompts, mainly focussed on spelling. Some helpful exports from this module are:
  - `create_icl_prompt()`
  - `spelling_formatter()`: formatter which outputs the spelling of a token
  - `first_letter_formatter()`: formatter which outputs the first letter of a token
- `sae_spelling.vocab`: Helpers for working with token-vocabularies. Some helpful exports from this module are:
  - `get_alpha_tokens()`: Filter tokens from tokenizer vocab which are alphabetic
- `sae_spelling.sae_utils`: Helpers for working with SAEs. Main exports include:
  - `apply_saes_and_run()`: Apply SAEs to a model and run on a prompt. Allows providing a list of hooks and optionally track activation gradients. This is used in attribution and ablation experiments.
- `sae_spelling.spelling_grader`: Code for grading spelling prompts. Some helpful exports from this module are:
  - `SpellingGrader`: Class for grading model performing on spelling prompts
- `sae_spelling.feature_absorption_calculator`: Code for calculating feature absorption. Some helpful exports from this module are:
  - `FeatureAbsorptionCalculator`

### Experiments

We include the following experiments from the paper in the `sae_spelling.experiments` package:

- `sae_spelling.experiments.latent_evaluation`: This experiment finds the top SAE latent for each first-letter spelling task, and evaluates the latent's performance relative to a LR probe.
- `sae_spelling.experiments.k_sparse_probing`: This experiment trains k-sparse probes on the first-letter task, and evaluates the performance with increasing value of `k`. This is used to detect feature splitting.
- `sae_spelling.experiments.feature_absorption`: This experiment attempts to quantify feature absorption on the first-letter task across SAEs.

These experiments each include a main "runner" function to run the experiment. These runners will only create data-frames and save them to disk, but won't generate plots. Experiments packages include helpers for generating the plots in the paper, but these plots require tex to be installed, so we don't generate plots by default.

**NOTE**
The experiments all require logistic-regression probes to be trained and data about the train/test split to be saved into dataframes in a specific format. We have not yet moved that code into this repo, but will do that in the next few days.

## Development

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, [Pyright](https://github.com/microsoft/pyright) for type checking, and [Pytest](https://docs.pytest.org/en/stable/) for testing.

To run all checks, run:

```
make check-ci
```

In VSCode, you can install the Ruff extension to get linting and formatting automatically in the editor. It's recommended to enable formatting on save.

You can install a pre-commit hook to run linting and type-checking before commit automatically with:

```
poetry run pre-commit install
```

### Poetry tips

Below are some helpful tips for working with Poetry:

- Install a new main dependency: `poetry add <package>`
- Install a new development dependency: `poetry add --dev <package>`
  - Development dependencies are not required for the main code to run, but are for things like linting/type-checking/etc...
- Update the lockfile: `poetry lock`
- Run a command using the virtual environment: `poetry run <command>`
- Run a Python file from the CLI as a script (module-style): `poetry run python -m sae_spelling.path.to.file`

## Citation

Please cite this work as follows:

```
@misc{chanin2024absorption,
      title={A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders},
      author={David Chanin and James Wilken-Smith and Tomáš Dulka and Hardik Bhatnagar and Joseph Bloom},
      year={2024},
      eprint={2409.14507},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.14507},
}
```
