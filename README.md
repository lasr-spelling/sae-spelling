# SAE Spelling

Shared code for SAE spelling experiments as part of LASR

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install dependencies, run:

```
poetry install
```

## Development

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, [Pyright](https://github.com/microsoft/pyright) for type checking, and [Pytest](https://docs.pytest.org/en/stable/) for testing.

To run all checks, run:

```
make check-ci
```

In VSCode, you can install the Ruff extension to get linting and formatting automatically in the editor. It's recommended to enable formatting on save.
