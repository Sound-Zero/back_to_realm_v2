"""Backward-compatible environment module.

The original project exposed `Env` from this file. New code should import
`Env_v2` from `env.env_v2`; `Env` remains as a compatibility alias.
"""

from env.env_v2 import Env_v2


class Env(Env_v2):
    pass
