.PHONY: venv tests

tests:
	mypy -p tests
	mypy -p growing_transformer
	pytest
