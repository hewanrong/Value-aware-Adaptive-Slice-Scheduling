from __future__ import annotations

from .knapsack import multiple_choice_knapsack


def schedule_oracle(items: list[dict], budget: int, qualities: list[str]) -> list[dict]:
    return multiple_choice_knapsack(items, budget, qualities)
