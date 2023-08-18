lint:
	poetry run black zrive-ds tests

test:
	poetry run flake8 src tests
	poetry run mypy src
	poetry run pytest tests
