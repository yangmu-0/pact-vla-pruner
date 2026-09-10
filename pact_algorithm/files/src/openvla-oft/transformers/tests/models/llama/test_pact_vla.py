import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[3] / "src" / "transformers" / "models" / "llama" / "pact_vla.py"
SPEC = importlib.util.spec_from_file_location("pact_vla", MODULE_PATH)
PACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACT)


def _scores(scores):
    return torch.tensor(scores, dtype=torch.float32)


def _controller(**overrides):
    settings = {
        "budget_rates": (0.25, 0.5, 1.0),
        "gamma": 0.8,
        "theta0": 0.6,
        "alpha_d": 0.0,
        "theta_min": 0.6,
        "theta_max": 1.0,
    }
    settings.update(overrides)
    return PACT.PACTController(**settings)


def test_method_threshold_defaults_are_current_values():
    controller = PACT.PACTController((0.25, 0.5, 1.0))
    assert controller.theta0 == 0.4
    assert controller.theta_min == 0.15
    assert controller.theta_max == 1.0


def test_cold_start_uses_perception_prior_and_selects_small_budget():
    selected, stats = _controller().select(_scores([4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), ())
    assert selected.numel() == 2
    assert stats["cold_start"] is True
    assert stats["d_t"] == pytest.approx(0.0)
    assert stats["fallback"] is False


def test_conflicting_routes_expand_budget():
    perception = _scores([4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = torch.tensor([0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    selected, stats = _controller().select(perception, (action,))
    assert selected.numel() == 4
    assert stats["d_t"] > 0.9
    assert stats["dual_cov"] >= stats["theta_t"]


def test_action_history_is_normalized_before_temporal_weighting():
    perception = PACT.normalize_distribution(torch.ones(2))
    old = torch.tensor([100.0, 0.0])
    recent = torch.tensor([0.0, 1.0])
    prior, cold = PACT.build_action_prior((old, recent), perception, gamma=0.5)
    assert cold is False
    assert prior.tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0], abs=1e-6)


def test_selection_keeps_highest_joint_priority_nominees():
    candidates = torch.tensor([0, 1, 2, 3])
    priority = torch.tensor([0.1, 0.9, 0.5, 0.3])
    assert PACT.select_top_priority(candidates, priority, budget=2).tolist() == [1, 2]


def test_selection_returns_sorted_candidates_when_budget_matches_union():
    candidates = torch.tensor([2, 0, 1])
    priority = torch.tensor([0.1, 0.9, 0.5])
    assert PACT.select_top_priority(candidates, priority, budget=3).tolist() == [0, 1, 2]


def test_each_step_uses_its_current_minimum_budget_immediately():
    perception = _scores([4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    controller = _controller()
    conflict = torch.tensor([0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    first, _ = controller.select(perception, (conflict,))
    second, _ = controller.select(perception, (perception,))
    assert first.numel() == 4
    assert second.numel() == 2


def test_invalid_attribution_returns_full_token_fallback():
    selected, stats = _controller().select(_scores([1.0, float("nan"), 0.0, 0.0]), ())
    assert selected.tolist() == [0, 1, 2, 3]
    assert stats["fallback"] is True
    assert "finite" in stats["fallback_reason"]
