.PHONY: install test smoke check build

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

smoke:
	dial-det smoke --contract dial-out-at-most
	dial-det smoke --contract dial-out-exact
	dial-det smoke --contract dial-prop-exact
	dial-det smoke --contract dial-prop-cost

check:
	python scripts/release_check.py

build:
	python -m build
