PYTHON ?= python3
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: setup test smoke

setup:
	$(PYTHON) -m pip install --index-url $(TORCH_INDEX) "torch>=2.4"
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q tests

smoke:
	PYTHON=$(PYTHON) bash smoke_test.sh
