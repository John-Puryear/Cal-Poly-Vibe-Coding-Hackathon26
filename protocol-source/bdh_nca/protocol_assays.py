"""Strict, model-independent screens for the synthetic planarian protocol.

Inputs are recurrent cell tensors, never a model, control, checkpoint, or
teacher. These screens are anatomical proxies; they cannot establish biological
identity from connected components alone. Training targets may be passed only
as the reference body mask and (for timing) the reference voltage/anatomy maps.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .metrics import components


@dataclass(frozen=True)
class ProtocolThresholds:
    tissue: float = 0.35
    head: float = 0.45
    eye: float = 0.35
    tail: float = 0.35
    body_iou: float = 0.80
    head_min_area: int = 10
    eye_min_area: int = 2
    tail_min_area: int = 2
    eye_pair_min_separation: float = 1.5
    eye_head_max_distance: float = 5.0
    eye_pair_max_midpoint_offset: float = 2.5
    left_end_fraction: float = 0.42
    right_end_fraction: float = 0.58


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _cells(value):
    result = _array(value)
    if result.ndim != 4 or result.shape[1] < 15:
        raise ValueError("Expected cells with shape (batch, >=15, height, width)")
    return result


def _body_reference(value, shape):
    if value is None:
        return None
    result = _array(value)
    if result.ndim == 4:
        result = result[:, 0]
    if result.ndim != 3 or result.shape != (shape[0], shape[2], shape[3]):
        raise ValueError("Reference body must have shape (batch, height, width) or (batch, channels, height, width)")
    return result > 0.35


def _paired_eyes(eyes, head, limits):
    """Require two substantial, vertically separated eyes near one head."""
    if len(eyes) != 2 or any(eye["area"] < limits.eye_min_area for eye in eyes):
        return False
    upper, lower = sorted(eyes, key=lambda eye: eye["y"])
    if lower["y"] - upper["y"] < limits.eye_pair_min_separation:
        return False
    if abs((upper["y"] + lower["y"]) / 2 - head["y"]) > limits.eye_pair_max_midpoint_offset:
        return False
    return all(np.hypot(eye["x"] - head["x"], eye["y"] - head["y"]) <= limits.eye_head_max_distance
               for eye in eyes)


def score_planaria(cells, target_body=None, limits=ProtocolThresholds()):
    """Return per-sample normal and two-head anatomical screens.

    The anterior is the left side of the grid, as in the synthetic generator.
    Supply ``target_body`` to enforce the body overlap gate. Without it, body
    overlap is reported as unavailable and only connectivity constrains shape.
    """
    array = _cells(cells)
    reference = _body_reference(target_body, array.shape)
    width = array.shape[-1]
    left_edge = limits.left_end_fraction * (width - 1)
    right_edge = limits.right_end_fraction * (width - 1)
    rows = []
    for index, sample in enumerate(array):
        body = sample[0] > limits.tissue
        body_parts = components(body, min_area=1)
        heads = components((sample[1] > limits.head) & body, min_area=3)
        eyes = components((sample[2] > limits.eye) & body, min_area=1)
        tails = components((sample[3] > limits.tail) & body, min_area=1)
        body_area = int(body.sum())
        body_connected = len(body_parts) == 1
        if reference is None:
            body_iou = None
            area_ratio = None
            shape_pass = body_connected and body_area > 0
        else:
            wanted = reference[index]
            union = int((body | wanted).sum())
            body_iou = float((body & wanted).sum() / max(1, union))
            area_ratio = float(body_area / max(1, int(wanted.sum())))
            shape_pass = body_connected and body_iou >= limits.body_iou

        left_heads = [head for head in heads if head["x"] < left_edge and head["area"] >= limits.head_min_area]
        right_heads = [head for head in heads if head["x"] > right_edge and head["area"] >= limits.head_min_area]
        left_eyes = [eye for eye in eyes if eye["x"] < left_edge]
        right_eyes = [eye for eye in eyes if eye["x"] > right_edge]
        good_left_eyes = len(left_heads) == 1 and _paired_eyes(left_eyes, left_heads[0], limits)
        good_right_eyes = len(right_heads) == 1 and _paired_eyes(right_eyes, right_heads[0], limits)
        normal = (shape_pass and len(heads) == 1 and len(left_heads) == 1 and
                  len(eyes) == 2 and good_left_eyes and len(tails) == 1 and
                  tails[0]["area"] >= limits.tail_min_area and tails[0]["x"] > right_edge)
        two_head = (shape_pass and len(heads) == 2 and len(left_heads) == 1 and
                    len(right_heads) == 1 and len(eyes) == 4 and good_left_eyes and
                    good_right_eyes and len(tails) == 0)
        rows.append({
            "normal_pass": bool(normal), "two_head_pass": bool(two_head),
            "body_area": body_area, "body_component_areas": [part["area"] for part in body_parts],
            "body_connected": body_connected, "body_iou": body_iou,
            "body_area_ratio": area_ratio, "head_count": len(heads),
            "eye_count": len(eyes), "tail_count": len(tails),
            "head_domains": heads, "eye_domains": eyes, "tail_domains": tails,
            "left_eye_pair_pass": bool(good_left_eyes),
            "right_eye_pair_pass": bool(good_right_eyes),
        })
    return rows


def score_paired_protocol(sham_washout, treated_washout, sham_recut,
                          treated_recut, target_body,
                          negative_controls: Mapping | None = None,
                          limits=ProtocolThresholds()):
    """Gate a sustained treatment against its matched sham at both endpoints.

    Optional ``negative_controls`` maps a name to ``(washout, recut)`` cell
    tensors. Each negative case must remain normal at both endpoints.
    """
    states = {
        "sham_washout": sham_washout, "treated_washout": treated_washout,
        "sham_recut": sham_recut, "treated_recut": treated_recut,
    }
    negatives = dict(negative_controls or {})
    for name, pair in negatives.items():
        if len(pair) != 2:
            raise ValueError(f"Negative control {name!r} needs (washout, recut) cells")
        states[f"{name}_washout"], states[f"{name}_recut"] = pair
    batch_size = _cells(sham_washout).shape[0]
    reference = _body_reference(target_body, _cells(sham_washout).shape)
    if reference is None:
        raise ValueError("A reference body is required for the strict paired protocol")
    scored = {name: score_planaria(cells, reference, limits) for name, cells in states.items()}
    if any(len(rows) != batch_size for rows in scored.values()):
        raise ValueError("All paired protocol states must have the same batch size")
    rows = []
    for index in range(batch_size):
        case = {name: scores[index] for name, scores in scored.items()}
        sham_ok = case["sham_washout"]["normal_pass"] and case["sham_recut"]["normal_pass"]
        negative_ok = all(case[f"{name}_washout"]["normal_pass"] and
                          case[f"{name}_recut"]["normal_pass"] for name in negatives)
        washout_ok = sham_ok and negative_ok and case["treated_washout"]["two_head_pass"]
        recut_ok = washout_ok and case["treated_recut"]["two_head_pass"]
        rows.append({"washout_pass": bool(washout_ok), "retained_after_recut": bool(recut_ok),
                     "matched_sham_pass": bool(sham_ok), "negative_controls_pass": bool(negative_ok),
                     "cases": case})
    return rows


def score_prepattern(timeline: Mapping[int, object], target,
                     threshold=0.5, initial_tolerance=1e-4):
    """Measure whether a target-like voltage field precedes later face fields.

    ``timeline`` maps measured ticks to raw cell tensors, including tick zero
    when initial unseeded status should be verified. A score of 0 is an empty
    field; 1 is an exact target match in the target body. Timing alone is only
    a correlation screen; a separate voltage-clamp ablation tests dependence.
    """
    if not timeline:
        raise ValueError("Prepattern timeline is empty")
    expected = _cells(target)
    ordered = sorted(timeline)
    fields = {"voltage": 4, "visible_head": 1, "visible_eyes": 2,
              "regulatory": 12, "genomic_axis": 13, "structure": 14}
    scores = {name: [] for name in fields}
    for tick in ordered:
        sample = _cells(timeline[tick])
        if sample.shape != expected.shape:
            raise ValueError("Every timeline tensor must match the target shape")
        body = expected[:, 0:1] > 0.35
        for name, channel in fields.items():
            wanted = expected[:, channel:channel+1]
            observed = sample[:, channel:channel+1]
            numerator = ((observed - wanted)**2 * body).sum(axis=(1, 2, 3))
            denominator = (wanted**2 * body).sum(axis=(1, 2, 3))
            scores[name].append(1 - numerator / np.maximum(denominator, 1e-9))
    initial_unseeded = None
    if 0 in timeline:
        first = _cells(timeline[0])
        initial_unseeded = np.max(np.abs(first[:, list(fields.values())]), axis=(1, 2, 3)) <= initial_tolerance
    rows = []
    for index in range(expected.shape[0]):
        first_crossing = {
            name: next((tick for tick, value in zip(ordered, values)
                        if value[index] >= threshold), None)
            for name, values in scores.items()
        }
        voltage_tick = first_crossing["voltage"]
        later = ("visible_head", "regulatory", "genomic_axis", "structure")
        ordering = (voltage_tick is not None and
                    all(first_crossing[name] is not None and voltage_tick < first_crossing[name]
                        for name in later))
        eyes_tick = first_crossing["visible_eyes"]
        full_face = (ordering and eyes_tick is not None and voltage_tick < eyes_tick and
                     initial_unseeded is not None and bool(initial_unseeded[index]))
        rows.append({
            "voltage_first_ordering": bool(ordering),
            "full_face_prepattern_pass": bool(full_face),
            "initial_unseeded": None if initial_unseeded is None else bool(initial_unseeded[index]),
            "first_crossing": first_crossing,
            "scores": {name: {tick: float(values[pos][index]) for pos, tick in enumerate(ordered)}
                       for name, values in scores.items()},
        })
    return rows
