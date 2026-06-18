from __future__ import annotations

from collections import defaultdict


def run_offline_simulation(
    edge_results: list[dict],
    cloud_results: list[dict],
    schedule_decisions: list[dict],
) -> dict:
    edge_by_frame = defaultdict(list)
    cloud_by_slice = defaultdict(list)
    action_by_slice = {d["slice_id"]: d["action"] for d in schedule_decisions}
    for row in edge_results:
        edge_by_frame[row["frame_id"]].append(row)
    for row in cloud_results:
        cloud_by_slice[row["slice_id"]].append(row)

    frame_results = defaultdict(list)
    uploaded = 0
    for frame_id, detections in edge_by_frame.items():
        for det in detections:
            action = action_by_slice.get(det.get("slice_id"), "none")
            if action == "none":
                frame_results[frame_id].append(det)
            else:
                uploaded += 1
                frame_results[frame_id].extend(cloud_by_slice.get(det.get("slice_id"), []))
    return {
        "metrics": {"num_frames": len(frame_results), "uploaded_slices": uploaded},
        "frame_results": dict(frame_results),
    }
