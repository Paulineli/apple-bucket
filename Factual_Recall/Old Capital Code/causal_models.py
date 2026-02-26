"""
Causal model for capital factual recall.

Template: "The capital of A is B." We ask the LLM to fill in B.
A = region (country / state / province), B = capital (expected answer).

Like the positional entity model: inputs A and B, computed raw_input and raw_output.
Interchange intervention: replace representation at position A with counterfactual's A';
success = model outputs counterfactual's B (raw_output).
"""

from typing import Any

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism, input_var

try:
    from .config import CapitalTaskConfig
except ImportError:
    from config import CapitalTaskConfig


def _compute_raw_input(t: CausalTrace, config: CapitalTaskConfig) -> str:
    """raw_input = prompt with A filled in (model will generate B)."""
    return config.prompt_template.format(A=t["A"])


def _compute_raw_output(t: CausalTrace) -> str:
    """raw_output = expected answer B (the capital)."""
    return t["B"]


def create_capital_model(config: CapitalTaskConfig) -> CausalModel:
    """
    Create the capital factual recall causal model.

    Variables:
    - A: input (region: country/state/province)
    - B: input (capital, the expected answer)
    - raw_input: computed prompt "The capital of {A} is "
    - raw_output: computed expected answer B

    Counterfactual: (A', B') different pair. Interchange at position A:
    replace base representation at A with counterfactual's representation at A'.
    Success = model outputs B' (counterfactual's raw_output).

    Args:
        config: CapitalTaskConfig with template and capital_pairs.

    Returns:
        CausalModel instance.
    """
    regions = config.regions
    capitals = config.capitals

    mechanisms: dict[str, Mechanism] = {}
    values: dict[str, Any] = {}

    # Inputs
    mechanisms["A"] = input_var(regions)
    values["A"] = regions

    mechanisms["B"] = input_var(capitals)
    values["B"] = capitals

    # Computed
    mechanisms["raw_input"] = Mechanism(
        parents=["A"],
        compute=lambda t: _compute_raw_input(t, config),
    )
    values["raw_input"] = None

    mechanisms["raw_output"] = Mechanism(
        parents=["B"],
        compute=_compute_raw_output,
    )
    values["raw_output"] = None

    model_id = "capital_factual_recall"
    return CausalModel(mechanisms, values, id=model_id)
