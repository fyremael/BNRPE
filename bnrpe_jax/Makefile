.PHONY: validate-all ci-validate test bench experiments fusion sweep gates

PYTHON ?= python
ARTIFACTS ?= artifacts

validate-all:
	$(PYTHON) scripts/run_validation_suite.py --output-root $(ARTIFACTS) --mode full --with-sweep

ci-validate:
	$(PYTHON) scripts/run_validation_suite.py --output-root $(ARTIFACTS)_ci --mode ci

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

bench:
	$(PYTHON) scripts/benchmark_overhead.py --output-dir $(ARTIFACTS)/benchmarks

experiments:
	$(PYTHON) scripts/run_experiment_tables.py --output-dir $(ARTIFACTS)/experiments

fusion:
	$(PYTHON) scripts/prototype_fused_paths.py --output-dir $(ARTIFACTS)/fusion

sweep:
	$(PYTHON) scripts/sweep_fusion_hybrid.py --output-dir $(ARTIFACTS)/fusion_sweep

gates:
	$(PYTHON) scripts/build_gate_report.py --output-dir $(ARTIFACTS)/governance
