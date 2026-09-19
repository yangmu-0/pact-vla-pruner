"""Training-free PACT-VLA controller implementing Method Eqs. (4)--(17)."""

import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


class PACTValidationError(ValueError):
    """Raised when a PACT input cannot be used safely."""


PACT_VARIANTS = ("full", "no-conflict", "perception-only", "last-action-prior")


def normalize_distribution(scores: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize a finite, non-negative attribution vector (Eq. 4)."""
    if not torch.is_tensor(scores):
        raise PACTValidationError("Attribution scores must be a tensor.")
    values = scores.detach().to(torch.float32).flatten()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise PACTValidationError("Attribution scores must be non-empty and finite.")
    if torch.any(values < 0):
        raise PACTValidationError("Attribution scores must be non-negative.")
    values = values + float(epsilon)
    total = values.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        raise PACTValidationError("Attribution scores cannot be normalized.")
    distribution = values / total
    if not torch.isfinite(distribution).all() or not torch.isclose(
        distribution.sum(), torch.ones((), device=distribution.device), atol=1e-5, rtol=1e-5
    ):
        raise PACTValidationError("Normalized attribution is invalid.")
    return distribution


def build_action_prior(
    action_history: Optional[Sequence[torch.Tensor]],
    perception: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, bool]:
    """Build the causal, time-decayed action prior from completed decisions (Eq. 5)."""
    if not 0.0 < float(gamma) <= 1.0:
        raise PACTValidationError("PACT gamma must be in (0, 1].")
    history = list(action_history or ())
    if not history:
        return perception.clone(), True

    normalized = []
    for scores in reversed(history):  # newest first: gamma**0 has the largest weight
        distribution = normalize_distribution(scores).to(device=perception.device)
        if distribution.shape != perception.shape:
            raise PACTValidationError("Historical action attribution does not match the original token coordinates.")
        normalized.append(distribution)
    weights = torch.tensor(
        [float(gamma) ** index for index in range(len(normalized))],
        dtype=perception.dtype,
        device=perception.device,
    )
    weights = weights / weights.sum()
    prior = torch.stack(normalized, dim=0).mul(weights[:, None]).sum(dim=0)
    return normalize_distribution(prior), False


def normalized_jsd(left: torch.Tensor, right: torch.Tensor) -> float:
    """Normalized Jensen--Shannon divergence in [0, 1] (Eqs. 6--8)."""
    if left.shape != right.shape or left.numel() == 0:
        raise PACTValidationError("JSD inputs must have the same non-zero shape.")
    eps = torch.finfo(left.dtype).eps
    midpoint = 0.5 * (left + right)
    left_kl = torch.sum(left * (torch.log(left.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps))))
    right_kl = torch.sum(right * (torch.log(right.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps))))
    value = (0.5 * (left_kl + right_kl) / math.log(2.0)).clamp(0.0, 1.0)
    if not torch.isfinite(value):
        raise PACTValidationError("JSD is not finite.")
    return float(value.item())


def compute_theta(conflict: float, theta0: float, alpha_d: float, theta_min: float, theta_max: float) -> float:
    """Compute the conflict-dependent coverage threshold (Eq. 9)."""
    values = (conflict, theta0, alpha_d, theta_min, theta_max)
    if not all(math.isfinite(float(value)) for value in values):
        raise PACTValidationError("PACT threshold parameters must be finite.")
    if not 0.0 <= theta_min <= theta_max <= 1.0:
        raise PACTValidationError("PACT thresholds must satisfy 0 <= min <= max <= 1.")
    return min(theta_max, max(theta_min, theta0 + alpha_d * conflict))


def build_candidate_budgets(num_tokens: int, rates: Sequence[float]) -> Tuple[int, ...]:
    """Build sorted hardware-supported token budgets and always include N (Eq. 10)."""
    if num_tokens <= 0:
        raise PACTValidationError("PACT requires at least one visual token.")
    parsed = tuple(float(rate) for rate in rates)
    if not parsed or any(not math.isfinite(rate) or rate <= 0.0 or rate > 1.0 for rate in parsed):
        raise PACTValidationError("PACT budget rates must be finite and in (0, 1].")
    budgets = {max(1, min(num_tokens, math.ceil(rate * num_tokens))) for rate in parsed}
    budgets.add(num_tokens)
    return tuple(sorted(budgets))


def top_b_union(perception: torch.Tensor, action_prior: torch.Tensor, budget: int) -> torch.Tensor:
    """Let both evidence branches nominate their top-B tokens (Eq. 11)."""
    budget = max(1, min(int(budget), perception.numel()))
    perception_top = torch.topk(perception, budget, sorted=False).indices
    action_top = torch.topk(action_prior, budget, sorted=False).indices
    return torch.unique(torch.cat((perception_top, action_top)), sorted=True)


def select_top_priority(
    candidates: torch.Tensor,
    joint_priority: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Keep the B highest joint-priority nominees (Eqs. 12--13).

    The nomination union from Eq. (11) is ranked by ``joint_priority`` -- the
    elementwise maximum of the perception and action attributions -- and truncated
    with a single ``topk``.  This is one vectorised pass rather than an O(B)
    sequence of tiny synchronised GPU operations.
    """
    num_tokens = joint_priority.numel()
    candidates = candidates.to(device=joint_priority.device, dtype=torch.long).flatten()
    if candidates.numel() < budget or torch.unique(candidates).numel() != candidates.numel():
        raise PACTValidationError("The nomination union cannot supply the requested budget.")
    if torch.any(candidates < 0) or torch.any(candidates >= num_tokens):
        raise PACTValidationError("A candidate token index is out of range.")
    if candidates.numel() == budget:
        return candidates.sort().values

    joint_scores = joint_priority.index_select(0, candidates)
    keep = torch.topk(joint_scores, int(budget)).indices
    selected = candidates.index_select(0, keep)
    if selected.numel() != budget or torch.unique(selected).numel() != budget:
        raise PACTValidationError("Joint-priority selection did not return exactly B unique tokens.")
    return selected.sort().values


def compute_dual_coverage(
    perception: torch.Tensor, action_prior: torch.Tensor, selected: torch.Tensor
) -> Tuple[float, float, float]:
    """Compute separate perception/action coverage and their minimum (Eqs. 14--16)."""
    if selected.numel() == 0 or torch.any(selected < 0) or torch.any(selected >= perception.numel()):
        raise PACTValidationError("A selected token index is invalid.")
    cov_per = float(perception.index_select(0, selected).sum().item())
    cov_act = float(action_prior.index_select(0, selected).sum().item())
    return cov_per, cov_act, min(cov_per, cov_act)


def search_minimum_budget(
    perception: torch.Tensor,
    action_prior: torch.Tensor,
    budgets: Sequence[int],
    theta: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return the first candidate satisfying minimum sufficient dual coverage (Eq. 17)."""
    joint_priority = torch.maximum(perception, action_prior)
    candidate_records = []
    for budget in budgets:
        if int(budget) == perception.numel():
            candidates = torch.arange(perception.numel(), device=perception.device)
            selected = candidates
        else:
            candidates = top_b_union(perception, action_prior, int(budget))
            selected = select_top_priority(candidates, joint_priority, int(budget))
        cov_per, cov_act, dual_cov = compute_dual_coverage(perception, action_prior, selected)
        passed = dual_cov >= theta
        candidate_records.append(
            {
                "B": int(budget),
                "union_size": int(candidates.numel()),
                "cov_per": cov_per,
                "cov_act": cov_act,
                "dual_cov": dual_cov,
                "pass": bool(passed),
            }
        )
        if passed:
            return selected, {"candidate_records": candidate_records, **candidate_records[-1]}
    raise PACTValidationError("No legal candidate budget satisfied dual coverage.")


class PACTController:
    """Stateless-per-step PACT budget selection following the current Method."""

    def __init__(
        self,
        budget_rates: Sequence[float],
        variant: str = "full",
        gamma: float = 0.8,
        theta0: float = 0.4,
        alpha_d: float = 0.10,
        theta_min: float = 0.15,
        theta_max: float = 1.0,
    ) -> None:
        self.rates = tuple(float(rate) for rate in budget_rates)
        self.variant = str(variant)
        self.gamma = float(gamma)
        self.theta0 = float(theta0)
        self.alpha_d = float(alpha_d)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        build_candidate_budgets(1, self.rates)
        if self.variant not in PACT_VARIANTS:
            raise PACTValidationError(f"Unknown PACT variant {self.variant!r}; expected one of {PACT_VARIANTS}.")
        compute_theta(0.0, self.theta0, self.alpha_d, self.theta_min, self.theta_max)
        if not 0.0 < self.gamma <= 1.0:
            raise PACTValidationError("PACT gamma must be in (0, 1].")
        self.reset()

    @property
    def signature(self) -> Tuple:
        return (self.rates, self.variant, self.gamma, self.theta0, self.alpha_d, self.theta_min, self.theta_max)

    def reset(self) -> None:
        self._step = 0

    def select(
        self,
        perception_scores: torch.Tensor,
        action_history: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run Eqs. (4)--(17), falling back to all N tokens on any invalid input."""
        self._step += 1
        num_tokens = int(perception_scores.numel()) if torch.is_tensor(perception_scores) else 0
        device = perception_scores.device if torch.is_tensor(perception_scores) else torch.device("cpu")
        history_size = len(action_history or ())
        try:
            perception = normalize_distribution(perception_scores)
            history = list(action_history or ())
            if self.variant == "perception-only":
                action_prior, cold_start = perception.clone(), True
                effective_history_size = 0
            elif self.variant == "last-action-prior":
                action_prior, cold_start = build_action_prior(history[-1:], perception, self.gamma)
                effective_history_size = min(1, len(history))
            else:
                action_prior, cold_start = build_action_prior(history, perception, self.gamma)
                effective_history_size = len(history)
            if self.variant == "no-conflict":
                # This branch removes the conflict module from the timed path;
                # the fixed base threshold is still range-clamped below.
                conflict = None
                threshold_conflict = 0.0
            elif self.variant == "perception-only":
                conflict = 0.0
                threshold_conflict = 0.0
            else:
                conflict = normalized_jsd(perception, action_prior)
                threshold_conflict = conflict
            theta = compute_theta(threshold_conflict, self.theta0, self.alpha_d, self.theta_min, self.theta_max)
            budgets = build_candidate_budgets(perception.numel(), self.rates)
            selected, search_stats = search_minimum_budget(perception, action_prior, budgets, theta)
            focus_trace = {}
            if os.environ.get("PACT_TRACE_FOCUS") == "1":
                budget = int(selected.numel())
                perception_focus = torch.topk(perception, budget, sorted=False).indices.sort().values
                action_focus = torch.topk(action_prior, budget, sorted=False).indices.sort().values
                overlap = perception_focus[torch.isin(perception_focus, action_focus)]
                focus_trace = {
                    "perception_focus_indices": perception_focus.detach().cpu().tolist(),
                    "action_focus_indices": action_focus.detach().cpu().tolist(),
                    "focus_overlap_indices": overlap.detach().cpu().tolist(),
                }
            return selected, {
                "step": self._step,
                "variant": self.variant,
                "N": perception.numel(),
                "B_star": int(selected.numel()),
                "retention_ratio": float(selected.numel() / perception.numel()),
                "d_t": conflict,
                "theta_t": theta,
                "cov_per": search_stats["cov_per"],
                "cov_act": search_stats["cov_act"],
                "dual_cov": search_stats["dual_cov"],
                "history_size": effective_history_size,
                "available_history_size": history_size,
                "union_size": search_stats["union_size"],
                "selected_indices": selected.detach().cpu().tolist(),
                "candidate_records": search_stats["candidate_records"],
                "cold_start": cold_start,
                "fallback": False,
                "fallback_reason": None,
                **focus_trace,
            }
        except Exception as error:
            selected = torch.arange(max(0, num_tokens), device=device, dtype=torch.long)
            return selected, {
                "step": self._step,
                "variant": self.variant,
                "N": num_tokens,
                "B_star": num_tokens,
                "retention_ratio": 1.0 if num_tokens else 0.0,
                "d_t": None,
                "theta_t": None,
                "cov_per": 1.0 if num_tokens else None,
                "cov_act": 1.0 if num_tokens else None,
                "dual_cov": 1.0 if num_tokens else None,
                "history_size": history_size,
                "union_size": num_tokens,
                "selected_indices": selected.detach().cpu().tolist(),
                "candidate_records": [],
                "cold_start": history_size == 0,
                "fallback": True,
                "fallback_reason": f"{type(error).__name__}: {error}",
            }
