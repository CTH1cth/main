"""Training-only auxiliary losses for the UCOD experiments."""

from .oed_loss import (
    build_aux_evidence_loss,
    compute_logit_gradient_diagnostics,
    finalize_oed_epoch,
    new_oed_epoch_accumulator,
    update_oed_epoch_accumulator,
    validate_oed_config,
)

__all__ = [
    "build_aux_evidence_loss",
    "compute_logit_gradient_diagnostics",
    "finalize_oed_epoch",
    "new_oed_epoch_accumulator",
    "update_oed_epoch_accumulator",
    "validate_oed_config",
]
