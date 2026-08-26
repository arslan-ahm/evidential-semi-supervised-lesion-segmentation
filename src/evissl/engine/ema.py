"""Exponential moving average of model parameters (the teacher).

The teacher is a temporal ensemble of the student: averaging weights over
training steps yields a model that is more accurate and, critically for this
project, better calibrated than any single student checkpoint. Its predictions
on unlabelled data are what the consistency term distils back into the student
(Tarvainen & Valpola, 2017).

Two details that are easy to get wrong and expensive to debug:

**Decay warm-up.** A fixed decay of 0.99 means the teacher needs a few hundred
steps to forget its random initialisation. During that window it is worse than
the student and the consistency term actively teaches noise. The decay is
therefore ramped as ``min(decay, (1 + step) / (10 + step))``, which starts near
zero (teacher tracks student closely) and approaches the target decay.

**Buffers are copied, not averaged.** BatchNorm running statistics are buffers.
Averaging them across a moving parameter set produces statistics that match no
actual network, so buffers are copied straight from the student instead. The
default backbone uses GroupNorm and has no such buffers - which is the cleaner
fix - but the U-Net baseline does, and it must not be silently mishandled.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    """Maintains an exponential moving average of another model's parameters.

    Args:
        model: The student. A deep copy is taken as the initial teacher.
        decay: Target EMA decay, typically 0.99-0.999.
        warmup: Enable the decay ramp described in the module docstring.

    Attributes:
        module: The teacher network. Always in eval mode with
            ``requires_grad=False``, so it can be used for inference directly.
        step_count: Number of updates applied.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99, warmup: bool = True) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.warmup = warmup
        self.step_count = 0

        self.module = copy.deepcopy(model)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def current_decay(self) -> float:
        """The decay actually applied at the current step."""
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.step_count) / (10.0 + self.step_count))

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        """Pull the teacher towards ``model``. Returns the decay used."""
        decay = self.current_decay()
        student_params = dict(model.named_parameters())
        for name, teacher_param in self.module.named_parameters():
            student_param = student_params[name]
            # lerp_(other, w) computes self + w * (other - self); with
            # w = 1 - decay this is exactly decay * self + (1 - decay) * other.
            teacher_param.lerp_(student_param.detach().to(teacher_param.dtype), 1.0 - decay)

        student_buffers = dict(model.named_buffers())
        for name, teacher_buffer in self.module.named_buffers():
            teacher_buffer.copy_(student_buffers[name].detach())

        self.step_count += 1
        return decay

    def state_dict(self) -> dict:
        """Teacher weights plus the update counter."""
        return {"module": self.module.state_dict(), "step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        """Restore from :meth:`state_dict` or from a bare module state dict."""
        if "module" in state and isinstance(state["module"], dict):
            self.module.load_state_dict(state["module"])
            self.step_count = int(state.get("step_count", 0))
        else:
            self.module.load_state_dict(state)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forward through the teacher, always without gradients."""
        with torch.no_grad():
            return self.module(*args, **kwargs)
