from __future__ import annotations


def multiple_choice_knapsack(items: list[dict], budget: int, qualities: list[str]) -> list[dict]:
    states: dict[int, tuple[float, list[dict]]] = {0: (0.0, [])}
    for item in items:
        next_states: dict[int, tuple[float, list[dict]]] = {}
        for used, (value, chosen) in states.items():
            for q in qualities:
                cost = int(item.get(f"bytes_{q}", 0)) if q != "none" else 0
                gain = float(item.get(f"V_{q}", 0.0)) if q != "none" else 0.0
                new_used = used + cost
                if new_used > budget:
                    continue
                action = {"frame_id": item["frame_id"], "slice_id": item["slice_id"], "action": q}
                candidate = (value + gain, chosen + [action])
                if new_used not in next_states or candidate[0] > next_states[new_used][0]:
                    next_states[new_used] = candidate
        states = next_states or states
    return max(states.values(), key=lambda x: x[0])[1] if states else []
