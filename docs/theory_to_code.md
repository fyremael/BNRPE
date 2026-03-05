# Theory To Code

This page ties mathematical objects to concrete implementation units.

## Parameterization

- Mathematical object: low-rank per-axis generators.
- Implementation: `bnrpe_jax/bnrpe/params.py`
  - `BNRPEParams`: stores `U`, `M_raw`, `axis_scale`, `alpha`.
  - `skew`: projects matrices to skew-symmetric form.
  - `init_params`: initializes low-rank factors and scaling terms.

## Rotor Construction

- Mathematical object: `R(P) = Cayley(G(P))`.
- Implementation: `bnrpe_jax/bnrpe/rotors.py`
  - `generator_lowrank`: assembles concatenated low-rank factorization.
  - `_apply_cayley_lowrank`: generic low-rank Cayley apply path.
  - `_apply_bnrpe_oneaxis_flat`: exact single-axis fast path.
  - `_apply_bnrpe_twoaxis_flat`: exact two-axis direct batched solve path.
  - `apply_bnrpe`: shape-safe public application entrypoint.

## Regularization

- Mathematical object: commutator budget surrogate.
- Implementation: `bnrpe_jax/bnrpe/regularizers.py`
  - `_lowrank_commutator_fro2`: efficient small-matrix trace computation.
  - `commutator_budget_penalty`: axis-pair summed penalty.

## Validation and Governance

- Scripted suite: `bnrpe_jax/scripts/run_validation_suite.py`
- Gate assembly: `bnrpe_jax/scripts/build_gate_report.py`
- Mandatory dual-axis benchmark profile: `dual_axis_non_degenerate`
