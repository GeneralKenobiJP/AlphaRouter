from collections import deque

import numpy as np
from gymnasium import Wrapper


def _make_python_safe(obj):
    """Convert common non-Python-native types into plain Python types."""
    if obj is None:
        return None
    if isinstance(obj, deque):
        return list(obj)
    if isinstance(obj, np.generic):
        return obj.item()                 # e.g. np.int64 -> int
    if isinstance(obj, np.ndarray):
        # convert arrays (any shape) into nested lists (works for empty arrays too)
        return obj.tolist()
    # try to convert lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_python_safe(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _make_python_safe(v) for k, v in obj.items()}
    # otherwise leave as-is (strings, numbers, custom objects)
    return obj


class NormalizeInfoWrapper(Wrapper):
    """
    Ensure each step() returns info as a dict with a single key 'env_info'
    whose value is a plain Python object (lists, dicts, scalars). This forces
    VectorEnv to aggregate into an object array and avoids shape-broadcasting errors.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        safe = _make_python_safe(info)
        return obs, reward, terminated, truncated, {"env_info": safe}

    def reset(self, **kwargs):
        # gymnasium reset returns (obs, info)
        obs, info = self.env.reset(**kwargs)
        safe = _make_python_safe(info)
        return obs, {"env_info": safe}

    @property
    def _state(self):
        # delegate to inner env
        # .env should exist because Wrapper provides it
        return getattr(self.env, '_state')
