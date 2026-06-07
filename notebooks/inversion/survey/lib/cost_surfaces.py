"""Cost-surface evaluators for the survey.

Pure functions: each takes a candidate state (or its forward-model outputs)
and a target observed LC, and returns a scalar cost. No optimisation
orchestration, no candidate-pool bookkeeping — those belong to the
experiment scripts.

Surfaces in scope for the survey:
  - surrogate_full_lc_mse  (already in lib.surrogate_eval.full_lc_mse)
  - surrogate_bright_mse   (already in lib.surrogate_eval.bright_mse)
  - alignment_cost         (the m103-era surface; needs careful re-derivation
                            under correct truth — see concept page)
  - lofi_peak_match        (count bright peaks in target hit within ±5
                            epochs of bright peaks in prediction)

The alignment cost and lofi peak-match are tagged `# TODO survey` because
their parent-project implementations sit inside `m103_hybrid.py` and
`lib/lc_compare.py` — both contraband per CLAUDE.md. The first survey
experiment that needs them should re-implement here from the documented
formula, NOT import from upstream.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .surrogate_eval import predict  # for type-checkers only


def surrogate_full_lc_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Re-export of lib.surrogate_eval.full_lc_mse. See that module."""
    from .surrogate_eval import full_lc_mse
    return full_lc_mse(predicted, target)


def surrogate_bright_mse(
    predicted: np.ndarray,
    target: np.ndarray,
    bright_threshold: float = 11.0,
) -> float:
    """Re-export of lib.surrogate_eval.bright_mse. See that module."""
    from .surrogate_eval import bright_mse
    return bright_mse(predicted, target, bright_threshold)


def surrogate_nll_residual(
    predicted: np.ndarray,
    target: np.ndarray,
    sigma_mag: float = 0.05,
    rescale: bool = True,
) -> np.ndarray:
    """Re-export of lib.surrogate_eval.nll_residual (RF25 Eq. 3). Opt-in; s078.

    See that module for the scope note on σ_k vs the ‖S‖/‖Ŝ‖ rescaling.
    """
    from .surrogate_eval import nll_residual
    return nll_residual(predicted, target, sigma_mag, rescale)


def surrogate_nll_cost(
    predicted: np.ndarray,
    target: np.ndarray,
    sigma_mag: float = 0.05,
    rescale: bool = True,
) -> float:
    """Re-export of lib.surrogate_eval.nll_cost (scalar mean NLL). Opt-in; s078."""
    from .surrogate_eval import nll_cost
    return nll_cost(predicted, target, sigma_mag, rescale)


def alignment_cost(*args, **kwargs) -> float:
    """The m103-era alignment cost surface.

    NOT YET IMPLEMENTED in the survey. The parent-project formula combines
    bright-peak alignment, glint-window matching, and a phase-anchor sweep;
    its full definition is in the m103_hybrid.py source. To use it here,
    re-derive from the documented formula and re-test against a known
    reference seed under correct truth — do not import the m103 source.

    See concepts/known_pathologies_to_revalidate.md → "alignment cost
    is anti-truth (m135)" for the buggy-era claim that needs re-checking.
    """
    raise NotImplementedError(
        "alignment_cost not yet ported into the survey. The first experiment "
        "needing it should re-derive from the documented formula in concepts/"
        "and re-test under correct truth, NOT import from m103_hybrid.py."
    )


def lofi_peak_match(
    pred_lofi: np.ndarray,
    target_lofi: np.ndarray,
    epoch_window: int = 5,
    bright_threshold: float = 11.0,
) -> int:
    """Count of bright target-LC peaks matched by prediction within ±epoch_window.

    NOT YET IMPLEMENTED. The parent-project version uses a more elaborate
    prominence-weighted match. First experiment to need it should
    implement with a clear docstring citing this concept-page entry.
    """
    raise NotImplementedError(
        "lofi_peak_match not yet ported. Re-implement here when needed; "
        "do not import from m103_hybrid.py / lib/lc_compare.py."
    )
