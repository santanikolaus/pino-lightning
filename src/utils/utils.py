from math import prod
from typing import List

import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig


def instantiate_loggers(logger_cfg) -> List:
    loggers = []
    if not logger_cfg:
        return loggers
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(instantiate(lg_conf))
    return loggers


def instantiate_callbacks(callbacks_cfg) -> List[L.Callback]:
    callbacks = []
    if not callbacks_cfg:
        return callbacks
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(instantiate(cb_conf))
    return callbacks


def count_tensor_params(tensor, dims=None):
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2 * n_params
    return n_params
