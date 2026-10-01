"""Turn a segmentation into structured building data: rooms, polygons, areas, adjacency, confidence."""
from __future__ import annotations

import cv2
import numpy as np

from .classes import CLASS_NAMES, ROOM_CLASSES, WALL


def _polygon(mask: np.ndarray, scale: float) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    return [[round(float(x) / scale, 1), round(float(y) / scale, 1)] for x, y in c.reshape(-1, 2)]


def extract_rooms(label_map: np.ndarray, probs: np.ndarray | None = None, entropy: np.ndarray | None = None,
                  scale: float = 1.0, min_area_px: int = 150, review_threshold: float = 0.8,
                  adjacency_px: int = 12, split_share: float = 0.3) -> dict:
    """
    label_map: H,W class ids at working resolution.
    probs:     C,H,W calibrated probabilities (optional; enables confidence scores).
    scale:     working resolution / original resolution; polygons and areas are reported
               in original-image pixels.
    Rooms with confidence below `review_threshold` are flagged for human review.
    Known limitation: two rooms of the SAME type joined through a door gap are counted as one.
    """
    room_classes = np.array(ROOM_CLASSES)
    rooms, masks = [], []

    def add_room(mask: np.ndarray, cls: int) -> None:
        room = {"id": len(rooms), "type": CLASS_NAMES[cls], "polygon": _polygon(mask, scale),
                "area_px": round(float(mask.sum()) / scale**2, 1)}
        if probs is not None:
            conf = float(probs[cls][mask].mean())
            room["confidence"] = round(conf, 3)
            room["needs_review"] = conf < review_threshold
        if entropy is not None:
            room["mean_entropy"] = round(float(entropy[mask].mean()), 3)
        rooms.append(room)
        masks.append(mask)

    # 1) Find enclosed spaces: connected regions of "any room class", separated by walls.
    #    This is robust to pixel-level class noise, which would otherwise fragment rooms.
    n, regions = cv2.connectedComponents(np.isin(label_map, room_classes).astype(np.uint8), connectivity=4)
    for r in range(1, n):
        region = regions == r
        if region.sum() < min_area_px:
            continue
        counts = np.array([(label_map[region] == c).sum() for c in room_classes])
        # 2) Vote on the room type (probability mass if available, otherwise pixel counts).
        votes = probs[room_classes][:, region].sum(axis=1) if probs is not None else counts
        # 3) If a second type holds a large share (e.g. two rooms joined through a door gap), split by class.
        big = [c for c, k in zip(room_classes, counts) if k >= min_area_px and k >= split_share * region.sum()]
        if len(big) > 1:
            for cls in big:
                m_n, parts = cv2.connectedComponents((region & (label_map == cls)).astype(np.uint8), connectivity=4)
                for i in range(1, m_n):
                    part = parts == i
                    if part.sum() >= min_area_px:
                        add_room(part, int(cls))
        else:
            add_room(region, int(room_classes[int(np.argmax(votes))]))

    # Adjacency: rooms are neighbours if separated by at most ~adjacency_px (about one wall thickness).
    kernel = np.ones((adjacency_px, adjacency_px), np.uint8)
    dilated = [cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in masks]
    for i, room in enumerate(rooms):
        room["adjacent_to"] = [j for j in range(len(rooms)) if j != i and (dilated[i] & dilated[j]).any()]

    total_room_area = sum(r["area_px"] for r in rooms) or 1.0
    for r in rooms:
        r["area_fraction"] = round(r["area_px"] / total_room_area, 3)

    by_type: dict[str, int] = {}
    for r in rooms:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1

    h, w = label_map.shape
    return {
        "image_size_px": [round(w / scale), round(h / scale)],
        "units": "pixels (no drawing scale applied)",
        "summary": {
            "room_count": len(rooms),
            "rooms_by_type": by_type,
            "rooms_flagged_for_review": sum(1 for r in rooms if r.get("needs_review")),
            "wall_pixel_fraction": round(float((label_map == WALL).mean()), 3),
        },
        "rooms": rooms,
    }
