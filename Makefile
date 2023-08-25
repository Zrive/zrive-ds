lint:
	poetry run black src tests

test:
	poetry run flake8 src tests
	poetry run pytest tests
