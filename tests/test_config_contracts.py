from PPO.conf.conf import Config


def test_feature_and_sample_dimensions_are_consistent():
    assert Config.FEATURE_LEN == sum(Config.FEATURES)
    assert Config.MLP_FEATURE_LEN == Config.FEATURE_LEN
    assert Config.ACTION_SPACE_SIZE == Config.ACTION_NUM + Config.TALENT_NUM
    assert Config.SAMPLE_DIM == sum(Config.DATA_SPLIT_SHAPE)
    assert Config.DATA_SPLIT_SHAPE[0] == Config.FEATURE_LEN
    assert Config.DATA_SPLIT_SHAPE[-1] == Config.ACTION_SPACE_SIZE
