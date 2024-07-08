# SAE Spelling

Shared code for SAE spelling experiments as part of LASR

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install dependencies, run:

```
poetry install
```

#### Poetry tips

Below are some helpful tips for working with Poetry:

- Install a new main dependency: `poetry add <package>`
- Install a new development dependency: `poetry add --dev <package>`
  - Development dependencies are not required for the main code to run, but are for things like linting/type-checking/etc...
- Update the lockfile: `poetry lock`
- Run a command using the virtual environment: `poetry run <command>`
- Run a Python file from the CLI as a script (module-style): `poetry run python -m sae_spelling.path.to.file`

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
