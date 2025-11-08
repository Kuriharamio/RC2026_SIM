from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_shape,
)


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_check(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        last_action = env.action_manager.action
    else:
        last_action = env.action_manager.get_term(action_name).raw_actions

    if torch.max(last_action) > 100:
        print(f"Warning! Action {torch.max(last_action)} is out of range")

    return last_action
