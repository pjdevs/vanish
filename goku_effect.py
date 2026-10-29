"""Goku teleport: forehead two-finger pose and vertical comic-style speed lines."""

from __future__ import annotations

import numpy as np
import cv2


HOLD_SECONDS = 1.0
TELEPORT_SECONDS = 0.95
# Fraction of the effect at which the person is fully gone (or back, in reverse).
VANISH_POINT = 0.5

# MediaPipe face mesh indices: mid-forehead and the two cheek edges.
FOREHEAD = 151
FACE_LEFT, FACE_RIGHT = 234, 454


def _finger_ratio(points: np.ndarray, pip: int, tip: int) -> float:
    wrist = points[0]
    return float(np.linalg.norm(points[tip] - wrist) / max(np.linalg.norm(points[pip] - wrist), 1e-6))


def salute_fingertips(points: np.ndarray) -> np.ndarray | None:
    """Return the index/middle fingertip midpoint when only those two are straight and together."""
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return None
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if palm_width < 1e-6:
        return None
    index_straight = _finger_ratio(points, 6, 8) > 1.12
    middle_straight = _finger_ratio(points, 10, 12) > 1.12
    ring_folded = _finger_ratio(points, 14, 16) < 1.08
    pinky_folded = _finger_ratio(points, 18, 20) < 1.12
    together = np.linalg.norm(points[8] - points[12]) < 0.6 * palm_width
    if index_straight and middle_straight and (ring_folded or pinky_folded) and together:
        return (points[8] + points[12]) / 2.0
    return None


def forehead_from_face(face, width: int, height: int) -> tuple[np.ndarray, float] | None:
    if len(face) <= max(FOREHEAD, FACE_LEFT, FACE_RIGHT):
        return None
    forehead = np.array((face[FOREHEAD].x * width, face[FOREHEAD].y * height))
    left = np.array((face[FACE_LEFT].x * width, face[FACE_LEFT].y * height))
    right = np.array((face[FACE_RIGHT].x * width, face[FACE_RIGHT].y * height))
    return forehead, float(np.linalg.norm(right - left))


def salute_on_forehead(hands, forehead: np.ndarray, face_width: float,
                       width: int, height: int) -> bool:
    for landmarks in hands:
        points = np.array([(point.x * width, point.y * height) for point in landmarks])
        tips = salute_fingertips(points)
        if tips is not None and np.linalg.norm(tips - forehead) < 0.55 * face_width:
            return True
    return False


def make_speed_lines(mask: np.ndarray, seed: int) -> list[tuple[int, float, float, float, int]]:
    """Pick vertical lines (x, phase, speed, length, thickness) across the silhouette."""
    points = cv2.findNonZero((mask > 0.3).astype(np.uint8))
    if points is None:
        return []
    x, _, w, _ = cv2.boundingRect(points)
    rng = np.random.default_rng(seed)
    count = max(14, w // 9)
    return [
        (int(rng.uniform(x - 10, x + w + 10)), float(rng.random()), float(rng.uniform(1.6, 3.2)),
         float(rng.uniform(0.25, 0.7)), int(rng.choice((1, 2, 2, 3, 4))))
        for _ in range(count)
    ]


def _bump(value: float, start: float, peak: float, end: float) -> float:
    if value <= start or value >= end:
        return 0.0
    if value < peak:
        return (value - start) / (peak - start)
    return (end - value) / (end - peak)


def render_teleport(frame: np.ndarray, background: np.ndarray, live_mask: np.ndarray,
                    line_mask: np.ndarray, lines, age: float, reverse: bool = False) -> np.ndarray:
    """Draw one frame of the teleport; reverse plays the reappearance."""
    progress = min(max(age / TELEPORT_SECONDS, 0.0), 1.0)
    if reverse:
        progress = 1.0 - progress
    height, width = frame.shape[:2]

    # The person fades out around the vanish point while the lines peak.
    fade = np.clip((progress - (VANISH_POINT - 0.2)) / 0.2, 0.0, 1.0)
    person_alpha = 1.0 - fade
    alpha = live_mask * person_alpha
    output = (frame.astype(np.float32) * alpha[:, :, None]
              + background.astype(np.float32) * (1.0 - alpha[:, :, None]))

    if lines:
        intensity = min(progress / 0.18, 1.0) * (1.0 - np.clip((progress - 0.6) / 0.4, 0.0, 1.0))
        points = cv2.findNonZero((line_mask > 0.3).astype(np.uint8))
        if points is not None and intensity > 0.01:
            _, y, _, h = cv2.boundingRect(points)
            layer = np.zeros((height, width), dtype=np.float32)
            span = h * 1.3
            for x, phase, speed, length, thickness in lines:
                offset = (phase + age * speed) % 1.0
                center = y + h + 0.15 * h - offset * span
                half = length * h / 2.0
                cv2.line(layer, (x, int(center - half)), (x, int(center + half)),
                         1.0, thickness, cv2.LINE_AA)
            region = cv2.dilate((line_mask > 0.2).astype(np.uint8),
                                np.ones((1, 25), np.uint8)).astype(np.float32)
            region = cv2.GaussianBlur(region, (0, 0), sigmaX=6.0)
            layer *= region * intensity
            glow = cv2.GaussianBlur(layer, (0, 0), sigmaX=5.0) * 1.6
            core = np.clip(layer, 0.0, 1.0)[:, :, None]
            halo = np.clip(glow, 0.0, 1.0)[:, :, None]
            output = output * (1.0 - halo * 0.7) + np.array((255, 220, 120), np.float32) * halo * 0.7
            output = output * (1.0 - core) + 255.0 * core

    flash = _bump(progress, VANISH_POINT - 0.12, VANISH_POINT - 0.02, VANISH_POINT + 0.08)
    if flash > 0:
        glow = cv2.GaussianBlur(line_mask, (0, 0), sigmaX=10.0)[:, :, None] * flash * 0.85
        output = output * (1.0 - glow) + 255.0 * glow
    return np.clip(output, 0, 255).astype(np.uint8)
