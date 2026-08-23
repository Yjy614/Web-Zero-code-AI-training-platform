"""训练策略预设单元测试。"""

from app.schemas.task import TrainConfig
from app.services.train_strategy import (
    DEFAULT_TRAIN_STRATEGY,
    normalize_train_strategy,
    resolve_train_hparams,
)


def test_normalize_unknown_falls_back():
    assert normalize_train_strategy(None) == DEFAULT_TRAIN_STRATEGY
    assert normalize_train_strategy("") == DEFAULT_TRAIN_STRATEGY
    assert normalize_train_strategy("nope") == DEFAULT_TRAIN_STRATEGY
    assert normalize_train_strategy("finetune") == "finetune"


def test_resolve_hparams_stable():
    hp = resolve_train_hparams("stable")
    assert hp["train_strategy"] == "stable"
    assert hp["lr0"] == 0.01
    assert hp["cos_lr"] is False
    assert hp["warmup_epochs"] == 3.0


def test_resolve_hparams_fast_and_finetune():
    fast = resolve_train_hparams("fast")
    assert fast["cos_lr"] is True
    assert fast["warmup_epochs"] == 1.0
    fine = resolve_train_hparams("finetune")
    assert fine["lr0"] == 0.001
    assert fine["cos_lr"] is True


def test_train_config_normalizes_strategy():
    cfg = TrainConfig(train_strategy="bogus")
    assert cfg.train_strategy == "stable"
    cfg2 = TrainConfig(train_strategy="fast")
    assert cfg2.train_strategy == "fast"
