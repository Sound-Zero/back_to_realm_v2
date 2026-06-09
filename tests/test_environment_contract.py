import pytest

pytest.importorskip("PIL")
pytest.importorskip("toml")

from env.env_v2 import Env_v2


def test_environment_reset_returns_observation_and_extra_info():
    env = Env_v2()

    obs, extra_info = env.reset()

    assert extra_info["result_code"] == 0
    assert "frame_state" in obs
    assert "score_info" in obs
    assert len(obs["map_info"]) == 11
    assert len(obs["map_info"][0]["values"]) == 11
