import pytest


def test_actor_forward_shapes():
    torch = pytest.importorskip("torch")
    from PPO.conf.conf import Config
    from PPO.model.model import NetworkModelActor

    model = NetworkModelActor()
    feature = torch.zeros(1, Config.FEATURE_LEN)
    legal_action = torch.ones(1, Config.ACTION_SPACE_SIZE)

    prob, value = model(feature, legal_action)

    assert prob.shape == (1, Config.ACTION_SPACE_SIZE)
    assert value.shape == (1, Config.VALUE_NUM)
    assert torch.allclose(prob.sum(dim=1), torch.ones(1), atol=1e-5)


def test_learner_format_data_splits_sample_tensor():
    torch = pytest.importorskip("torch")
    from PPO.conf.conf import Config
    from PPO.model.model import NetworkModelLearner

    model = NetworkModelLearner()
    batch = torch.zeros(2, Config.SAMPLE_DIM)

    parts = model.format_data(batch)

    assert len(parts) == len(Config.DATA_SPLIT_SHAPE)
    assert [part.shape[1] for part in parts] == Config.DATA_SPLIT_SHAPE
