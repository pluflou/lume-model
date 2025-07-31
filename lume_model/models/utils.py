from typing import Union, Dict

from pydantic import BaseModel, ConfigDict
import torch
from torch.distributions import Distribution
import numpy as np


def _flatten_and_itemize(value):
    if isinstance(value, torch.Tensor):
        return [v.item() for v in value.flatten()]
    elif isinstance(value, np.ndarray):
        return [v.item() if hasattr(v, "item") else v for v in value.flatten()]
    else:
        return [value]


def itemize_dict(
    d: dict[str, Union[float, torch.Tensor, np.ndarray, Distribution]],
) -> list[dict[str, Union[float, torch.Tensor, np.ndarray]]]:
    """
    Converts a dictionary of values (floats, numpy arrays, or torch tensors) into a flat list of dictionaries,
    each containing the key-value pairs for the scalar elements in the original arrays/tensors.
    If the input dictionary contains only scalars (no arrays/tensors), returns a list with the original dict.

    Args:
        d: Dictionary to itemize. Values can be floats, numpy arrays, or torch tensors.

    Returns:
        List of dictionaries, each with the keys and scalar values from the original input, or the original dict in a list if no arrays/tensors are present.
    """
    has_arrays = any(
        isinstance(value, (torch.Tensor, np.ndarray)) for value in d.values()
    )
    itemized_dicts = []
    if has_arrays:
        for k, v in d.items():
            flat = _flatten_and_itemize(v)
            for i, ele in enumerate(flat):
                if i >= len(itemized_dicts):
                    itemized_dicts.append({k: ele})
                else:
                    itemized_dicts[i][k] = ele
    else:
        itemized_dicts = [d]
    return itemized_dicts


def format_inputs(
    input_dict: dict[str, Union[float, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Formats values of the input dictionary as tensors.

    Args:
        input_dict: Dictionary of input variable names to values.

    Returns:
        Dictionary of input variable names to tensors.
    """
    formatted_inputs = {}
    for var_name, value in input_dict.items():
        v = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        formatted_inputs[var_name] = v
    return formatted_inputs


class InputDictModel(BaseModel):
    """Pydantic model for input dictionary validation.

    Attributes:
        input_dict: Input dictionary to validate.
    """

    input_dict: Dict[str, Union[torch.Tensor, float]]

    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
