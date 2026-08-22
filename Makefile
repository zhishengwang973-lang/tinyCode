.PHONY: test compile check

PYTHON ?= python3

test:
	$(PYTHON) -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall -q tinyCode tests

check: test compile
