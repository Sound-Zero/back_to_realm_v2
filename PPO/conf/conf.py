"""Project configuration for the PPO training pipeline."""


class Config:
    """Static training and feature dimensions.

    The project keeps these values in a Python class because the original
    training code imports `Config` directly from many modules.
    """

    UPDATE_FREQ = 100
    LOAD_FREQ = 15
    SAVE_FREQ = 600
    CAPACITY = 100_000
    PRELOAD_MODEL_ID = "latest"

    GAMMA = 0.988
    TDLAMBDA = 0.96

    START_LR = 2e-4
    END_LR = 8e-6
    EDN_LR = END_LR  # Backward-compatible spelling used by the old code.

    BETA_START = 0.01
    CLIP_PARAM = 0.25
    VF_COEF = 0.67

    ACTION_LEN = 1
    ACTION_NUM = 8
    TALENT_NUM = 8
    ACTION_SPACE_SIZE = ACTION_NUM + TALENT_NUM

    feature_type = 2
    FEATURES = [
        2,          # current position
        5 * 11 * 11,
        11 * 11,
        11 * 11,
        6,          # end position
        6,          # historical position
        13,         # treasure state
        13 * 6,     # treasure positions
        8,          # buff feature
        3,          # talent feature
        1,          # time ratio
    ]

    MAP_FEATURE = 6 * 11 * 11
    MAP_FEATRUE = MAP_FEATURE  # Backward-compatible spelling.
    MLP_FEATURE_LEN = sum(FEATURES)
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    VALUE_NUM = 1

    DATA_SPLIT_SHAPE = [
        FEATURE_LEN,
        VALUE_NUM,              # reward
        VALUE_NUM,              # value
        VALUE_NUM,              # td lambda return
        VALUE_NUM,              # advantage
        ACTION_LEN,             # sampled action
        ACTION_LEN,             # sampled action probability
        ACTION_SPACE_SIZE,      # legal action mask
    ]
    data_len = sum(DATA_SPLIT_SHAPE)
    SAMPLE_DIM = data_len
