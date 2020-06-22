import os
import sys
import pickle
import config.settings

# create settings object corresponding to specified env
ENV = os.environ.get('ENV', 'Dev')
_current = getattr(sys.modules['config.settings'], '{0}Config'.format(ENV))()

# copy attributes to the module for convenience
for atr in [f for f in dir(_current) if not '__' in f]:
    # environment can override anything
    val = os.environ.get(atr, getattr(_current, atr))
    setattr(sys.modules[__name__], atr, val)


def as_dict():
    res = {}
    for atr in [f for f in dir(config) if not '__' in f]:
        val = getattr(config, atr)
        if type(val) in [float, int, list, dict, tuple, str, bool]:
            res[atr] = val
    return res
