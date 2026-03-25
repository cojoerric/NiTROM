---
name: architecture_preference
description: User prefers composition over duplication — wrapper classes should delegate to existing classes rather than copying their logic
type: feedback
---

When a new class wraps or extends an existing one (e.g., GasPolynomialModel wrapping PolynomialModel), compose by instantiating the inner class rather than duplicating its methods (evaluate_rhs, evaluate_adjoint_rhs, etc.).

**Why:** Avoids code duplication and ensures changes to the base class propagate automatically.

**How to apply:** When asked to create a "GAS" or "constrained" variant of an existing model, hold an instance of the base model internally and delegate forward/adjoint/etc. calls to it after assembling the constrained tensors.
