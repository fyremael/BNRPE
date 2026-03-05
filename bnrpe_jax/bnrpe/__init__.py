"""Public API for the BNR-PE JAX reference package."""

from .params import BNRPEParams, init_params
from .rotors import apply_bnrpe, apply_bnrpe_batch, attention_logits, generator_lowrank
from .regularizers import commutator_budget_penalty

__all__ = [
    "BNRPEParams",
    "init_params",
    "apply_bnrpe",
    "apply_bnrpe_batch",
    "attention_logits",
    "generator_lowrank",
    "commutator_budget_penalty",
]
