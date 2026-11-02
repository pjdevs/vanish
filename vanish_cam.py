"""Local webcam: dissolve triggered by a V sign with index and middle fingers."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .goku_effect import (
        HOLD_SECONDS, TELEPORT_SECONDS, VANISH_POINT, forehead_from_face,
        make_speed_lines, render_teleport, salute_on_forehead,
    )
    from .naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from .vanish_sound import SOUND_PATH, VanishSound
except ImportError:
    from goku_effect import (
        HOLD_SECONDS, TELEPORT_SECONDS, VANISH_POINT, forehead_from_face,
        make_speed_lines, render_teleport, salute_on_forehead,
    )
    from naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from vanish_sound import SOUND_PATH, VanishSound


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = APP_DIR / "face_landmarker.task"
# Drop an assets/teleport.MP3 in to replace the bundled synthetic sound.
TELEPORT_SOUND = next(
    (path for path in (APP_DIR / "assets" / "teleport.MP3", APP_DIR / "assets" / "teleport.wav")
     if path.is_file()),
    APP_DIR / "assets" / "teleport.wav",
)

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 3.6
GESTURE_INTERVAL_SECONDS = 0.14


GOKU_RELEASE_SECONDS = 0.3


@dataclass
class GokuState:
    hold_started: float | None = None
    lost_since: float | None = None
    needs_release: bool = False
    last_countdown: int | None = None
    mask: np.ndarray | None = None
    mask_at: float = float("-inf")
    line_mask: np.ndarray | None = None
    lines: list = field(default_factory=list)
    gain: np.ndarray | None = None
    last_report: float = 0.0


@dataclass
class AppState:
    phase: str = "calibrating"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


MODES_HELP = """modes:
  normal  Raise a V sign (index and middle fingers): you dissolve in ~3.6 s.
  naruto  Press your palms together: smoke cloud and a wooden log replace you.
  goku    Hold index and middle fingers together on your forehead for 1 s to
          teleport away; lower your hand and repeat to come back.

Step out of the frame for the first 4 seconds so the background is captured.

keys (in the preview window):
  SPACE  capture a new background   R  reappear (not in goku)
  V      trigger the effect (not in goku)   F  fullscreen   Q/ESC  quit
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam vanishing effects, sent to the OBS virtual camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MODES_HELP,
    )
    parser.add_argument("mode", nargs="?", choices=("normal", "naruto", "goku"),
                        help="Effect to run (see modes below)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=960, help="Video width")
    parser.add_argument("--height", type=int, default=540, help="Video height")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start in fullscreen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the preview window (the virtual camera is never mirrored)"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum FPS of the virtual output (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual webcam; by default it is enabled when available. --no-virtual-camera disables it",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Virtual webcam driver to use",
    )
    parser.add_argument("--virtual-device", help="Name of the virtual device to use")
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print recognition status in the terminal (no text in the video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode is None and not args.self_test:
        parser.error("choose a mode: normal, naruto or goku")
    return args


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        if platform.system() == "Darwin":
            hint = (
                "Install OBS Studio 30 or later, open it and press 'Start Virtual "
                "Camera' once to install the extension (approve it in "
                "System Settings), then stop it and quit OBS. "
            )
        else:
            hint = (
                "Install OBS Studio (with the OBS Virtual Camera driver) or "
                "Unity Capture. If you use OBS Virtual Camera, stop 'Start Virtual "
                "Camera' in OBS. "
            )
        message = (
            f"Virtual webcam not available. {hint}"
            "Start with 'uv run vanish_cam.py' to install pyvirtualcam. "
            f"Details: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Continuing with the local preview.")
        return None
    print(f"[Webcam] Virtual output active: {camera.device} ({width}x{height}, {args.fps:g} FPS max)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode == "naruto" and destination == GESTURE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[Vanish Lab] Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Vanish-Lab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe models. "
                "Check your internet connection and try again."
            ) from error


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} not available. Close other apps or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def base_options(model: Path) -> python.BaseOptions:
    # CPU only: on macOS the GPU (Metal) path leaks one frame per inference
    # (~2 MB at 960x540) until CoreVideo can no longer allocate and MediaPipe aborts.
    return python.BaseOptions(model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU)


def to_mp_image(frame: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def create_tasks(mode: str = "normal") -> tuple[vision.ImageSegmenter, MultiViewGestureRecognizer | MultiViewHandLandmarker]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=base_options(SEGMENTER_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode == "goku":
        return create_goku_tasks(segmenter_options)
    if mode == "naruto":
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Hand model not found: {HAND_MODEL}")
        options = vision.HandLandmarkerOptions(
            base_options=base_options(HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        task_class = vision.HandLandmarker
        wrapper_class = MultiViewHandLandmarker
    else:
        classifier_options = mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        )
        options = vision.GestureRecognizerOptions(
            base_options=base_options(GESTURE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=classifier_options,
        )
        task_class = vision.GestureRecognizer
        wrapper_class = MultiViewGestureRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = task_class.create_from_options(options)
        try:
            alternate = task_class.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    return segmenter, wrapper_class(primary, alternate)


def create_goku_tasks(segmenter_options) -> tuple[vision.ImageSegmenter, GokuDetector]:
    for model in (HAND_MODEL, FACE_MODEL):
        if not model.is_file():
            raise RuntimeError(f"Model not found: {model}")
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options(HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options(FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    with ExitStack() as cleanup:
        segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
        cleanup.callback(segmenter.close)
        hands = vision.HandLandmarker.create_from_options(hand_options)
        cleanup.callback(hands.close)
        face = vision.FaceLandmarker.create_from_options(face_options)
        cleanup.pop_all()
    return segmenter, GokuDetector(hands, face)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], background[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8)
        + cyan.astype(np.float32) * glow * 0.8,
        0,
        255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


def victory_landmark_score(points: np.ndarray) -> float:
    """Recognize extended index and middle fingers using 3D distances, not orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip: index, middle, ring, little.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Seen from the side, folded fingers can look straight but very short:
    # don't require clearly visible joints for the ring and little fingers.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two tips must be apart; the thumb may stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Geometric quality of the extended fingers, not a classifier probability.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # Landmarks are returned even when the classifier does not recognize
    # "Victory" (for example with the back of the hand facing the webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # ALWAYS check the visible geometry too: the estimated depth
    # can be inaccurate when the palm and folded fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Retry detection on rotated frames without disturbing tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE forces a fresh detection; the VIDEO model stays on the original
        # frame. Avoids inconsistent timestamps/tracking across rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # In the sideways pose the palm is often too small in the full
        # frame. The two lower crops overlap and cover both hands,
        # even when the image is mirrored.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=image.image_format,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=image.image_format,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class GokuDetector:
    """Track hands and face; remember the forehead while a hand covers it."""

    FOREHEAD_MEMORY_MS = 1500

    def __init__(self, hands, face):
        self.hands = hands
        self.face = face
        self.forehead: tuple[np.ndarray, float] | None = None
        self.forehead_at = 0
        self.last_hand_count = 0

    def detect_salute(self, image: mp.Image, timestamp_ms: int) -> bool:
        hands = self.hands.detect_for_video(image, timestamp_ms).hand_landmarks
        faces = self.face.detect_for_video(image, timestamp_ms).face_landmarks
        self.last_hand_count = len(hands)
        if faces:
            found = forehead_from_face(faces[0], image.width, image.height)
            if found is not None:
                self.forehead = found
                self.forehead_at = timestamp_ms
        if self.forehead is None or timestamp_ms - self.forehead_at > self.FOREHEAD_MEMORY_MS:
            return False
        return salute_on_forehead(hands, *self.forehead, image.width, image.height)

    def close(self) -> None:
        try:
            self.hands.close()
        finally:
            self.face.close()


def exposure_gain(frame: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel factor that brings the background to the live exposure, measured away from the person."""
    keep = cv2.dilate((mask > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8))[::8, ::8] == 0
    if keep.sum() < 200:
        return np.ones(3, np.float32)
    live = np.median(frame[::8, ::8][keep], axis=0)
    fixed = np.median(background[::8, ::8][keep], axis=0)
    return np.clip(live / np.maximum(fixed, 1.0), 0.5, 2.0).astype(np.float32)


def match_exposure(background: np.ndarray, gain: np.ndarray) -> np.ndarray:
    return np.clip(background.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def update_goku(
    state: AppState,
    goku: GokuState,
    frame: np.ndarray,
    background: np.ndarray,
    now: float,
    timestamp_ms: int,
    segmenter: vision.ImageSegmenter,
    detector: GokuDetector,
    sound: VanishSound | None,
    debug: bool,
) -> np.ndarray:
    """Hold the forehead pose for HOLD_SECONDS to teleport out, and again to come back."""
    height, width = frame.shape[:2]
    if state.phase in ("ready", "vanished") and now - state.last_gesture_check >= 0.08:
        state.last_gesture_check = now
        image = to_mp_image(frame)
        salute = detector.detect_salute(image, timestamp_ms)
        if debug and now - goku.last_report >= 1.0:
            print(f"[Goku] hands={detector.last_hand_count}, "
                  f"face={'yes' if detector.forehead is not None else 'no'}, pose={salute}")
            goku.last_report = now
        if not salute:
            # A short tracking dropout neither breaks the countdown nor counts as lowering the hand.
            if goku.lost_since is None:
                goku.lost_since = now
            if now - goku.lost_since >= GOKU_RELEASE_SECONDS:
                if goku.hold_started is not None:
                    print("[Goku] Pose broken, countdown reset.")
                goku.hold_started = None
                goku.needs_release = False
        elif goku.needs_release:
            goku.lost_since = None
        else:
            goku.lost_since = None
            if goku.hold_started is None:
                goku.hold_started = now
                goku.last_countdown = None
            remaining = max(0.0, HOLD_SECONDS - (now - goku.hold_started))
            step = int(np.ceil(remaining * 4))
            if step != goku.last_countdown and remaining > 0:
                print(f"[Goku] Hold it... {remaining:.2f} s")
                goku.last_countdown = step
            if remaining <= 0:
                goku.hold_started = None
                goku.needs_release = True
                leaving = state.phase == "ready"
                state.phase = "goku_out" if leaving else "goku_in"
                state.vanish_started = now
                goku.mask = person_mask(segmenter, image, timestamp_ms, width, height)
                goku.mask_at = now
                goku.line_mask = goku.mask
                goku.lines = make_speed_lines(goku.mask, seed=timestamp_ms)
                goku.gain = exposure_gain(frame, background, goku.mask)
                if sound is not None:
                    sound.play()
                print("[Goku] Instant transmission!" if leaving
                      else "[Goku] Back! Lower your hand before teleporting again.")

    if state.phase == "ready":
        return frame
    if now - goku.mask_at >= 0.05:
        goku.mask = person_mask(segmenter, to_mp_image(frame), timestamp_ms, width, height)
        goku.mask_at = now
        if state.phase == "goku_in" or (
            state.phase == "goku_out" and now - state.vanish_started < VANISH_POINT * TELEPORT_SECONDS
        ):
            # Lines follow the person while they are visible.
            goku.line_mask = goku.mask
        # The webcam's auto exposure drifts with the person in or out of frame:
        # follow it so the frozen background keeps matching the live room.
        goku.gain += 0.2 * (exposure_gain(frame, background, goku.mask) - goku.gain)
    scene = match_exposure(background, goku.gain)
    if state.phase == "vanished":
        return scene

    age = now - state.vanish_started
    reverse = state.phase == "goku_in"
    if age >= TELEPORT_SECONDS:
        state.phase = "ready" if reverse else "vanished"
        state.vanish_started = None
        return frame if reverse else scene
    return render_teleport(frame, scene, goku.mask, goku.line_mask, goku.lines, age, reverse)


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Start the dissolve on the first reliable V recognition."""
    if state.phase == "ready" and name == "Victory" and score >= 0.62:
        state.phase = "vanishing"
        state.vanish_started = now
        return True
    return False


def start_naruto_on_join(state: AppState, joined: bool, now: float) -> bool:
    if state.phase == "ready" and joined:
        state.phase = "naruto_vanishing"
        state.vanish_started = now
        return True
    return False


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH, TELEPORT_SOUND)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    load_log()
    if platform.system() != "Darwin":
        import pygame.mixer  # The distributed build must contain its MP3 decoder.
    for mode in ("normal", "naruto", "goku"):
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] Self-test complete: models, assets and audio present.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    with ExitStack() as resources:
        return run_camera(args, resources, args.mode)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str) -> int:
    download_models(mode)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)
    log_sprite = load_log() if mode == "naruto" else None
    if mode == "naruto":
        vanish_sound = VanishSound()
    elif mode == "goku":
        vanish_sound = VanishSound(TELEPORT_SOUND)
    else:
        vanish_sound = None
    if vanish_sound is not None:
        resources.callback(vanish_sound.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(
        calibration_started=time.perf_counter(), fullscreen=args.fullscreen
    )
    goku = GokuState()
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    log_anchor: tuple[int, int, int] | None = None
    log_cache: dict[int, np.ndarray] = {}
    vanished_frame: np.ndarray | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = 0.0
    started = time.perf_counter()
    print("[Webcam] Step out of the frame for 4 seconds: capturing the background.")
    print("[Webcam] Window controls: SPACE new background, R reappear, V vanish, F fullscreen, Q quit.")

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")

            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            if mode == "normal" and (dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
            now = time.perf_counter()

            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Capturing background: {remaining} seconds.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = "ready"
                    print("[Webcam] Background ready. You can step back into the frame.")
                display = camera_frame

            elif background is None:
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            elif mode == "goku":
                display = update_goku(
                    state, goku, camera_frame, background, now, elapsed_ms,
                    segmenter, recognizer, vanish_sound, args.gesture_debug,
                )

            elif state.phase == "vanished":
                if vanished_frame is None:
                    vanished_frame = background.copy()
                    if mode == "naruto" and log_anchor is not None:
                        overlay_log(vanished_frame, log_sprite, landed_log_anchor(log_anchor, height),
                                    cache=log_cache)
                display = vanished_frame

            else:
                mask = last_mask
                if (
                    state.phase == "ready"
                    and now - state.last_gesture_check >=
                    (0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)
                ):
                    state.last_gesture_check = now
                    mp_image = to_mp_image(camera_frame)
                    if mode == "normal":
                        name, score = recognize_gesture(recognizer, mp_image, elapsed_ms)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(
                                f"[Gesture] {name}, score={score:.2f}, "
                                f"crop={recognizer.preferred_crop or 'full'}, "
                                f"rotation={recognizer.preferred_rotation or 0}"
                            )
                            last_gesture_report = now
                        start_vanish_on_gesture(state, name, score, now)
                    else:
                        joined = recognizer.recognize_joined_hands(mp_image, elapsed_ms)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(f"[Hands] detected={recognizer.last_hand_count}, "
                                  f"joined={joined}, view={recognizer.preferred_view or 'full'}")
                            last_gesture_report = now
                        if start_naruto_on_join(state, joined, time.perf_counter()):
                            vanish_sound.play()

                if state.phase in ("vanishing", "naruto_vanishing"):
                    # Normal dissolve tracks the silhouette at 20 Hz; Naruto
                    # needs it only while the person is still visible.
                    needs_mask = (
                        (mode == "normal" and (mask is None or now - last_mask_at >= 0.05))
                        or (mode == "naruto" and (log_anchor is None or now - state.vanish_started < 0.13))
                    )
                    if needs_mask:
                        if mp_image is None:
                            mp_image = to_mp_image(camera_frame)
                        mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        last_mask = mask
                        last_mask_at = now
                    if mode == "naruto" and log_anchor is None:
                        log_anchor = substitution_anchor(mask)

                if mode == "naruto" and state.phase == "naruto_vanishing":
                    age = max(0.04, time.perf_counter() - state.vanish_started)
                    display = render_substitution(
                        camera_frame, background, mask, log_sprite, log_anchor, age, log_cache
                    )
                    if age >= EFFECT_SECONDS:
                        state.phase = "vanished"
                        vanished_frame = display
                elif state.phase == "ready":
                    display = camera_frame
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    display = apply_dissolve(
                        camera_frame, background, mask, progress, dissolve_field, sparkle_noise
                    )
                    if state.phase == "vanished":
                        vanished_frame = display

            if virtual_camera is not None:
                # Only the preview is mirrored: call apps mirror the self-view themselves
                # and must send the true orientation to everyone else.
                send_virtual_frame(virtual_camera,
                                   display if args.no_mirror else cv2.flip(display, 1))
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                goku = GokuState()
                state.phase = "calibrating"
                state.calibration_started = now
                state.vanish_started = None
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] Step out of the frame for 4 seconds: capturing the new background.")
            elif mode == "goku" and key in (ord("r"), ord("v")):
                pass  # The gesture is the only way to teleport out and back.
            elif key == ord("r"):
                if background is not None:
                    state.phase = "ready"
                    state.vanish_started = None
                    vanished_frame = None
                    log_anchor = None
                    last_mask = None
                    last_mask_at = float("-inf")
            elif key == ord("v") and background is not None:
                state.phase = "naruto_vanishing" if mode == "naruto" else "vanishing"
                state.vanish_started = time.perf_counter()
                if mode == "naruto":
                    vanish_sound.play()
                    log_anchor = None
                vanished_frame = None
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[Vanish Lab] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
