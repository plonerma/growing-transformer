.PHONY: venv test

test:
	mypy -p tests
	mypy -p growing_transformer
