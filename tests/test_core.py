"""Fast unit tests (no trained model or dataset needed).  Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpintel.calibration import fit_temperature, reliability  # noqa: E402
from fpintel.classes import NUM_CLASSES, WALL  # noqa: E402
from fpintel.metrics import iou_from_confusion, update_confusion  # noqa: E402
from fpintel.postprocess import extract_rooms  # noqa: E402


def two_room_plan():
    m = np.zeros((100, 200), np.uint8)
    m[10:90, 10:95] = 5    # bedroom
    m[10:90, 105:190] = 3  # kitchen
    m[10:90, 95:105] = WALL
    return m


def test_extract_rooms_counts_types_and_adjacency():
    out = extract_rooms(two_room_plan(), min_area_px=50)
    assert out["summary"]["room_count"] == 2
    assert out["summary"]["rooms_by_type"] == {"Kitchen": 1, "Bedroom": 1}
    assert out["rooms"][0]["adjacent_to"] == [1] and out["rooms"][1]["adjacent_to"] == [0]
    assert len(out["rooms"][0]["polygon"]) >= 4


def test_scale_maps_back_to_original_pixels():
    out = extract_rooms(two_room_plan(), min_area_px=50, scale=0.5)
    assert out["image_size_px"] == [400, 200]
    assert abs(out["rooms"][0]["area_px"] - 80 * 85 * 4) < 1


def test_low_confidence_room_is_flagged():
    m = two_room_plan()
    probs = np.full((NUM_CLASSES, *m.shape), 0.01, np.float32)
    probs[5][m == 5] = 0.95
    probs[3][m == 3] = 0.55
    out = extract_rooms(m, probs, min_area_px=50, review_threshold=0.8)
    flags = {r["type"]: r["needs_review"] for r in out["rooms"]}
    assert flags == {"Kitchen": True, "Bedroom": False}


def test_temperature_scaling_recovers_overconfidence():
    torch.manual_seed(0)
    labels = torch.randint(0, 5, (20000,))
    true_logits = torch.randn(20000, 5)
    true_logits[torch.arange(20000), labels] += 1.5
    # Sample labels from the true distribution, then make the model 3x overconfident.
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    t = fit_temperature(true_logits * 3.0, labels)
    assert 2.5 < t < 3.5
    ece_before, _ = reliability(torch.softmax(true_logits * 3, 1).numpy(), labels.numpy())
    ece_after, _ = reliability(torch.softmax(true_logits * 3 / t, 1).numpy(), labels.numpy())
    assert ece_after < ece_before


def test_iou_perfect_prediction():
    m = two_room_plan()
    conf = update_confusion(np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64), m, m)
    miou, _ = iou_from_confusion(conf)
    assert miou == 1.0


def test_pixel_noise_does_not_fragment_rooms():
    m = two_room_plan()
    rng = np.random.default_rng(0)
    noise = (rng.random(m.shape) < 0.1) & (m == 5)
    m[noise] = 6  # 10% of bedroom pixels mislabelled as bath
    out = extract_rooms(m, min_area_px=50)
    assert out["summary"]["rooms_by_type"] == {"Bedroom": 1, "Kitchen": 1}


def test_rooms_joined_by_door_gap_are_split_by_type():
    m = two_room_plan()
    m[40:60, 95:105] = 5  # door gap: bedroom pixels bridge into the kitchen side
    out = extract_rooms(m, min_area_px=50)
    assert out["summary"]["room_count"] == 2
