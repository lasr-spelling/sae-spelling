format:
	poetry run ruff format

check-linting:
	poetry run ruff check
	poetry run ruff format --check

check-type:
	poetry run pyright .

test:
	poetry run pytest

check-ci:
	make check-linting
	make check-type
	make test
