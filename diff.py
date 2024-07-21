import torch
from torch import nn


def diff_model_state(new_model: nn.Module, old_model: nn.Module) -> dict:
    """

    """
    state_dict_new = new_model.state_dict()
    state_dict_old = old_model.state_dict()
    diff_state = {}
    for k, v in state_dict_old.items():
        if torch.equal(v, state_dict_new[k]):
            continue
        #  Tracking the layers which have same keys and different values.
        if k in state_dict_new:
            diff_state[k] = v
    for k, v in state_dict_new.items():
        # if new_model has more layers than the old one, then these layers should be marked and removed
        # when it comes to patching stage. Here I just use a tricky method that assign them with a special value '#D'.
        if k not in state_dict_old:
            diff_state[k] = "#D"
    return diff_state


def patch_model_state(current_model: nn.Module, original_model: nn.Module, diff_state: dict) -> nn.Module:
    current_model_state_dict = current_model.state_dict()
    for k, v in diff_state.items():
        if isinstance(v, str) and v == "#D":
            del current_model_state_dict[k]
            del diff_state[k]

    current_model_state_dict.update(diff_state)
    original_model.load_state_dict(current_model_state_dict)
    return original_model
