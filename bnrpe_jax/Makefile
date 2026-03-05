.PHONY: validate-all test bench experiments fusion

PYTHON ?= python
ARTIFACTS ?= artifacts

validate-all:
	$(PYTHON) -m pytest -q -p no:cacheprovider
	$(PYTHON) scripts/benchmark_overhead.py --output-dir $(ARTIFACTS)/benchmarks
	$(PYTHON) scripts/run_experiment_tables.py --output-dir $(ARTIFACTS)/experiments
	$(PYTHON) scripts/prototype_fused_paths.py --output-dir $(ARTIFACTS)/fusion

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

bench:
	$(PYTHON) scripts/benchmark_overhead.py --output-dir $(ARTIFACTS)/benchmarks

experiments:
	$(PYTHON) scripts/run_experiment_tables.py --output-dir $(ARTIFACTS)/experiments

fusion:
	$(PYTHON) scripts/prototype_fused_paths.py --output-dir $(ARTIFACTS)/fusion
