# Motivation

BNR-PE explores non-commutative, low-rank positional transforms that preserve vector norms through exact Cayley maps.

## Why This Matters

- Standard positional encodings often constrain axis interactions.
- Multi-axis tasks can benefit from controlled non-Abelian behavior.
- Exact orthogonality avoids positional drift in magnitude-sensitive paths.

## Core Hypothesis

Represent each axis with a structured low-rank skew generator and combine them through a rotor map:

- Axis generator: `A_a = U_a * skew(M_a) * U_a^T`
- Position-conditioned generator: `G(P) = sum_a c_a(P) * A_a`
- Rotor: `R(P) = Cayley(G(P))`

This gives a tractable route to richer geometry while preserving norms.

## Practical Constraints

- Runtime overhead is regime-dependent (`d`, `L`, `r`), not uniform.
- Low-width, higher-rank dual-axis cases remain the hardest regime.
- Governance gates enforce both quality and overhead constraints.
