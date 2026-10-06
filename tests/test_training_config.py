"""Конфиги: базовый + метод + оверлеи вариантов; переопределения пространства условий; val варианта."""
import numpy as np
import pytest

from quadwind.conditions import TRAIN_SPACE, U, make_suite, override_space
from quadwind.training import build_space, load_config


def test_overlay_merges_and_names_run():
    cfg = load_config("configs/rppo.yaml", ["configs/e7/v4_all.yaml"])
    assert cfg["name"] == "rppo__v4_all" and cfg["algo"] == "rppo"
    assert cfg["env"]["sigma_near"] == 0.05 and cfg["algo_kwargs"]["n_steps"] == 512
    space = build_space(cfg)
    assert space.jump_rate.bounds(1.0) == (0.0, 0.0) and space.noise_scale.bounds(1.0) == (1.0, 2.5)
    assert space.noise_scale.bounds(0.0) == (0.5, 0.5)            # curriculum задан явно
    assert space.slow_tau.bounds(1.0) == (10.0, 20.0)
    assert space.mass_ratio == TRAIN_SPACE.mass_ratio              # не тронутое — как в train
    assert load_config("configs/rppo.yaml")["name"] == "rppo"


def test_override_keeps_curriculum_and_rejects_unknown_keys():
    space = override_space(TRAIN_SPACE, {"wind_speed": [0.0, 5.0], "act_delay": [0, 2], "turb_type": "dryden"})
    assert space.wind_speed.bounds(0.0) == (0.0, 0.0) and space.wind_speed.bounds(1.0) == (0.0, 5.0)
    assert space.act_delay.bounds(1.0) == (0, 2) and space.turb_type == "dryden"
    with pytest.raises(AttributeError):
        override_space(TRAIN_SPACE, {"no_such_param": 1})


def test_variant_val_suite_uses_same_seeds_other_distribution():
    space = override_space(TRAIN_SPACE, {"noise_scale": [2.0, 2.0]})
    base, variant = make_suite("val", 8), make_suite("val", 8, space=space)
    assert [c.seed for c in base] == [c.seed for c in variant]
    assert all(c.noise_scale == 2.0 for c in variant) and not all(c.noise_scale == 2.0 for c in base)
    assert np.isclose(U(1, 1).sample(np.random.default_rng(0)), 1.0)
