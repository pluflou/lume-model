"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

For now, only scalar floating-point variables are supported.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Tuple, Literal, Callable, Union
from enum import Enum
import math

import numpy as np
from numpy.typing import NDArray
import torch
from torch.distributions import Distribution as TDistribution
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from pydantic_core import core_schema


class ConfigEnum(str, Enum):
    """Enum for configuration options during validation."""

    NULL = "none"
    WARN = "warn"
    ERROR = "error"


class Variable(BaseModel, ABC):
    """Abstract variable base class.

    Attributes:
        name: Name of the variable.
    """

    name: str

    @property
    @abstractmethod
    def default_validation_config(self) -> ConfigEnum:
        """Determines default behavior during validation."""
        return None

    @abstractmethod
    def validate_value(self, value: Any, config: dict[str, bool] = None):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        return {"variable_class": self.__class__.__name__} | config


class ScalarVariable(Variable):
    """Variable for float values.

    Attributes:
        default_value: Default value for the variable. Note that the LUMEBaseModel requires this
          for input variables, but it is optional for output variables.
        value_range: Value range that is considered valid for the variable. If the value range is set to None,
          the variable is interpreted as a constant and values are validated against the default value.
        is_constant: Flag indicating whether the variable is constant.
        unit: Unit associated with the variable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_value: Optional[float] = None
    is_constant: Optional[bool] = False
    value_range: Optional[tuple[float, float]] = None
    # tolerance for floating point errors, currently only used for constant variables
    value_range_tolerance: Optional[float] = 1e-8
    unit: Optional[str] = None

    @field_validator("value_range", mode="before")
    @classmethod
    def validate_value_range(cls, value):
        if value is not None:
            value = tuple(value)
            if not value[0] <= value[1]:
                raise ValueError(
                    f"Minimum value ({value[0]}) must be lower or equal than maximum ({value[1]})."
                )
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None and self.value_range is not None:
            if not self._value_is_within_range(self.default_value):
                raise ValueError(
                    "Default value ({}) is out of valid range: ([{},{}]).".format(
                        self.default_value, *self.value_range
                    )
                )
        return self

    @model_validator(mode="after")
    def validate_constant_value_range(self):
        if self.is_constant and self.value_range is not None:
            # if the upper limit is not equal to the lower limit, raise an error
            if not self.value_range[0] == self.value_range[1]:
                error_message = (
                    f"Expected range to be constant for constant variable '{self.name}', "
                    f"but received a range of values. Set range to None or set the "
                    f"upper limit equal to the lower limit."
                )
                raise ValueError(error_message)
        return self

    @property
    def default_validation_config(self) -> ConfigEnum:
        return "warn"

    def validate_value(self, value: float, config: ConfigEnum = None):
        """
        Validates the given value.

        Args:
            value (float): The value to be validated.
            config (ConfigEnum, optional): The configuration for validation. Defaults to None.
              Allowed values are "none", "warn", and "error".

        Raises:
            TypeError: If the value is not of type float.
            ValueError: If the value is out of the valid range or does not match the default value
              for constant variables.
        """
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        self._validate_value_type(value)
        # optional validation
        if config != "none":
            self._validate_value_is_within_range(value, config=_config)

    @staticmethod
    def _validate_value_type(value: float):
        if not isinstance(value, float):
            raise TypeError(
                f"Expected value to be of type {float} or {np.float64}, "
                f"but received {type(value)}."
            )

    def _validate_value_is_within_range(self, value: float, config: ConfigEnum = None):
        if not self._value_is_within_range(value):
            if self.is_constant:
                error_message = "Expected value to be ({}) for constant variable ({}), but received ({}).".format(
                    self.default_value, self.name, value
                )
            else:
                error_message = (
                    "Value ({}) of '{}' is out of valid range: ([{},{}]).".format(
                        value, self.name, *self.value_range
                    )
                )
            range_warning_message = (
                error_message
                + " Executing the model outside of the training data range may result in"
                " unpredictable and invalid predictions."
            )
            if config == "warn":
                print("Warning: " + range_warning_message)
            else:
                raise ValueError(error_message)

    def _value_is_within_range(self, value) -> bool:
        self.value_range = self.value_range or (-np.inf, np.inf)
        tolerances = {"rel_tol": 0, "abs_tol": self.value_range_tolerance}
        is_within_range, is_within_tolerance = False, False
        # constant variables
        if self.value_range is None or self.is_constant:
            if self.default_value is None:
                is_within_tolerance = True
            else:
                is_within_tolerance = math.isclose(
                    value, self.default_value, **tolerances
                )
        # non-constant variables
        else:
            is_within_range = self.value_range[0] <= value <= self.value_range[1]
            is_within_tolerance = any(
                [math.isclose(value, ele, **tolerances) for ele in self.value_range]
            )
        return is_within_range or is_within_tolerance


class DistributionVariable(Variable):
    """Variable for distributions. Must be a subclass of torch.distributions.Distribution.

    Attributes:
        unit: Unit associated with the variable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unit: Optional[str] = None

    @property
    def default_validation_config(self) -> ConfigEnum:
        return "none"

    def validate_value(self, value: TDistribution, config: ConfigEnum = None):
        """
        Validates the given value.

        Args:
            value (Distribution): The value to be validated.
            config (ConfigEnum, optional): The configuration for validation. Defaults to None.
              Allowed values are "none", "warn", and "error".

        Raises:
            TypeError: If the value is not an instance of Distribution.
        """
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        self._validate_value_type(value)
        # optional validation
        if config != "none":
            pass  # not implemented

    @staticmethod
    def _validate_value_type(value: TDistribution):
        if not isinstance(value, TDistribution):
            raise TypeError(
                f"Expected value to be of type {TDistribution}, "
                f"but received {type(value)}."
            )


class NumpyNDArray(np.ndarray):
    """Custom Pydantic-compatible type for numpy.ndarray."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(
        cls,
        value: Any,
        expected_shape: Tuple[int, ...] = None,
        expected_dtype: np.dtype = None,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Value must be a numpy.ndarray, got {type(value)}")
        if expected_shape and tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {tuple(value.shape)}"
            )
        if expected_dtype and value.dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype}, got {value.dtype}")
        return value


class TorchTensor(torch.Tensor):
    """Custom Pydantic-compatible type for torch.Tensor."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(
        cls,
        value: Any,
        expected_shape: Tuple[int, ...] = None,
        expected_dtype: torch.dtype = None,
        expected_device: str = None,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Value must be a torch.Tensor, got {type(value)}")
        if expected_shape and tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {tuple(value.shape)}"
            )
        if expected_dtype and value.dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype}, got {value.dtype}")
        if expected_device and value.device.type != expected_device:
            raise ValueError(
                f"Expected device {expected_device}, got {value.device.type}"
            )
        return value


class ArrayVariable(Variable):
    """
    Variable for array data (NumpyNDArray or TorchTensor).

    Attributes:
        default_value: Default value for the variable.
        name: Name of the variable.
        shape: Shape of the array.
        dtype: Data type of the array.
        unit: Unit associated with the variable.
        array_type: Type of array, either 'numpy' or 'torch'.
        device: Device for torch.Tensor ('cpu' or 'cuda'), optional.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    default_value: Optional[Union[NDArray, torch.Tensor]] = None
    shape: Tuple[int, ...]
    dtype: Any
    unit: Optional[str] = None
    array_type: Literal["numpy", "torch"] = "torch"
    device: Optional[str] = None

    @property
    def default_validation_config(self):
        return "warn"

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            if self.array_type == "numpy":
                if isinstance(self.default_value, np.ndarray):
                    NumpyNDArray.validate(
                        self.default_value,
                        expected_shape=self.shape,
                        expected_dtype=self.dtype,
                    )
                else:
                    raise TypeError(
                        f"Expected default_value to be a numpy.ndarray, "
                        f"but received {type(self.default_value)}."
                    )
            elif self.array_type == "torch":
                if isinstance(self.default_value, torch.Tensor):
                    TorchTensor.validate(
                        self.default_value,
                        expected_shape=self.shape,
                        expected_dtype=self.dtype,
                        expected_device=self.device,
                    )
                else:
                    raise TypeError(
                        f"Expected default_value to be a torch.Tensor, "
                        f"but received {type(self.default_value)}."
                    )
        return self

    def validate_value(
        self, value: Union[NumpyNDArray, TorchTensor], config: str = None
    ):
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        print(value)
        if self.array_type == "numpy":
            NumpyNDArray.validate(
                value,
                expected_shape=self.shape,
                expected_dtype=self.dtype,
            )
        elif self.array_type == "torch":
            TorchTensor.validate(
                value,
                expected_shape=self.shape,
                expected_dtype=self.dtype,
                expected_device=self.device,
            )
        # optional validation
        if config != "none":
            pass  # TODO: implement optional validation logic, like range checks, checking for NaNs, etc.


class ImageVariable(ArrayVariable):
    """
    Variable for image data (2D or 3D numpy arrays or torch tensors).

    Enforces that shape is 2D or 3D and allows for image-specific metadata.
    """

    num_channels: Optional[int] = None  # 1 (grayscale), or 3 (RGB))

    def validate_value(self, value: Any, config: ConfigEnum = None):
        # First, use ArrayVariable validation
        super().validate_value(value, config)
        # Then, enforce image shape
        if len(self.shape) not in (2, 3):
            raise ValueError(
                f"ImageVariable expects shape to be 2D or 3D, got {self.shape}."
            )
        if len(value.shape) not in (2, 3):
            raise ValueError(
                f"Value for ImageVariable must be 2D or 3D, got {value.shape}."
            )

        # TODO: leave section below or delete?
        if self.num_channels is not None:
            # Check for numpy arrays and torch tensors
            if self.array_type == "numpy":
                if len(value.shape) == 2 and self.num_channels != 1:
                    raise ValueError(
                        f"Expected 1 channel for grayscale image, got {self.num_channels}."
                    )
                elif len(value.shape) == 3 and value.shape[2] != self.num_channels:
                    raise ValueError(
                        f"Expected {self.num_channels} channels, got {value.shape[2]}."
                    )
            elif self.array_type == "torch":
                if len(value.shape) == 2 and self.num_channels != 1:
                    raise ValueError(
                        f"Expected 1 channel for grayscale image, got {self.num_channels}."
                    )
                elif len(value.shape) == 3 and value.shape[0] != self.num_channels:
                    raise ValueError(
                        f"Expected {self.num_channels} channels, got {value.shape[0]}."
                    )


def get_variable(name: str) -> Type[Variable]:
    """Returns the Variable subclass with the given name.

    Args:
        name: Name of the Variable subclass.

    Returns:
        Variable subclass with the given name.
    """
    classes = [ScalarVariable, DistributionVariable, ArrayVariable, ImageVariable]
    class_lookup = {c.__name__: c for c in classes}
    if name not in class_lookup.keys():
        raise KeyError(
            f"No variable named {name}, valid names are {list(class_lookup.keys())}"
        )
    return class_lookup[name]
