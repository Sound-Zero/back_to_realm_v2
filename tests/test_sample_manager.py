import numpy as np

from PPO.conf.conf import Config
from PPO.feature.definition import SampleManager


def test_sample_manager_packs_expected_width():
    manager = SampleManager()
    feature = np.zeros(Config.FEATURE_LEN, dtype=np.float32)
    legal_action = np.ones(Config.ACTION_SPACE_SIZE, dtype=np.float32)

    manager.sample_process(
        feature=feature,
        legal_action=legal_action,
        prob=np.array([1.0], dtype=np.float32),
        action=np.array([0], dtype=np.float32),
        value=np.array([0.0], dtype=np.float32),
        reward=np.array([0.1], dtype=np.float32),
    )
    manager.process_last_frame(np.array([0.0], dtype=np.float32))

    samples = manager.get_game_data()
    assert len(samples) == 1
    assert samples[0].npdata.shape == (Config.SAMPLE_DIM,)
    assert samples[0].npdata.dtype == np.float32
