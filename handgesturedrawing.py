import hashlib
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import mediapipe as mp
import numpy as np


# ---------- Settings ----------

CAMERA_INDEX = 0
BRUSH_COLOR = (0, 255, 0)
BRUSH_THICKNESS = 5

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SIZE_BYTES = 7_819_105
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


# ---------- Model helpers ----------

def get_file_hash(path):
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def is_model_valid(path):
    """Check whether the model file is the expected version."""
    return (
        path.is_file()
        and path.stat().st_size == MODEL_SIZE_BYTES
        and get_file_hash(path) == MODEL_SHA256
    )


def download_model():
    """Download the MediaPipe model and verify its integrity."""
    print("Downloading MediaPipe hand model...")

    temporary_path = None

    try:
        request = Request(MODEL_URL, headers={"User-Agent": "AirDrawing/1.0"})

        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=MODEL_PATH.parent,
                prefix=".hand_model_",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)

        if not is_model_valid(temporary_path):
            raise RuntimeError("Downloaded model failed validation.")

        temporary_path.replace(MODEL_PATH)
        temporary_path = None
        print("Model downloaded successfully.")

    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Could not download the hand model.\nDetails: {error}"
        ) from error

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def ensure_model_exists():
    """Use the existing valid model or download it once."""
    if MODEL_PATH.exists():
        if is_model_valid(MODEL_PATH):
            return

        raise RuntimeError(
            f"Invalid model found: {MODEL_PATH}\n"
            "Delete the file and run the program again."
        )

    download_model()


# ---------- Setup helpers ----------

def open_camera():
    """Open the webcam with a Windows-friendly fallback."""
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    return camera


def create_hand_detector():
    """Create the MediaPipe hand detector."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(options)


# ---------- Hand detection helpers ----------

def landmark_to_pixel(landmark, width, height):
    """Convert a normalized landmark into screen coordinates."""
    x = int(np.clip(landmark.x, 0.0, 1.0) * (width - 1))
    y = int(np.clip(landmark.y, 0.0, 1.0) * (height - 1))
    return x, y


def is_finger_extended(landmarks, mcp_id, pip_id, tip_id):
    """Check whether a finger is straight."""
    mcp = np.array((landmarks[mcp_id].x, landmarks[mcp_id].y))
    pip = np.array((landmarks[pip_id].x, landmarks[pip_id].y))
    tip = np.array((landmarks[tip_id].x, landmarks[tip_id].y))

    lower_segment = pip - mcp
    upper_segment = tip - pip

    denominator = np.linalg.norm(lower_segment) * np.linalg.norm(upper_segment)

    if denominator < 1e-6:
        return False

    alignment = np.dot(lower_segment, upper_segment) / denominator
    return alignment > 0.75


def is_drawing_gesture(landmarks):
    """Draw when index is extended and middle finger is folded."""
    index_extended = is_finger_extended(landmarks, 5, 6, 8)
    middle_extended = is_finger_extended(landmarks, 9, 10, 12)

    return index_extended and not middle_extended


def draw_hand(frame, landmarks):
    """Draw the detected hand skeleton."""
    height, width = frame.shape[:2]
    points = [
        landmark_to_pixel(landmark, width, height)
        for landmark in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 1)

    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


# ---------- Display helpers ----------

def draw_status(output, status, height):
    """Show the current app state and controls."""
    colors = {
        "DRAWING": BRUSH_COLOR,
        "PAUSED": (0, 165, 255),
        "NO HAND": (0, 0, 255),
    }

    cv2.putText(
        output,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        colors[status],
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "Index up + middle folded: draw | C: clear | Q/Esc: quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def process_hand(frame, canvas, result, previous_point):
    """Process the detected hand and update the drawing state."""
    height, width = frame.shape[:2]

    if not result.hand_landmarks:
        return "NO HAND", None

    landmarks = result.hand_landmarks[0]
    draw_hand(frame, landmarks)

    if not is_drawing_gesture(landmarks):
        return "PAUSED", None

    current_point = landmark_to_pixel(landmarks[8], width, height)
    if previous_point is not None:
        cv2.line(
            canvas,
            previous_point,
            current_point,
            BRUSH_COLOR,
            BRUSH_THICKNESS,
            cv2.LINE_AA,
        )

    cv2.circle(frame, current_point, 7, BRUSH_COLOR, -1, cv2.LINE_AA)
    return "DRAWING", current_point


def next_timestamp(last_timestamp_ms):
    """Return a strictly increasing timestamp for video-mode detection."""
    timestamp_ms = time.monotonic_ns() // 1_000_000
    return max(timestamp_ms, last_timestamp_ms + 1)


def main():
    ensure_model_exists()
    camera = open_camera()

    canvas = None
    previous_point = None
    last_timestamp_ms = -1

    try:
        with create_hand_detector() as detector:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                height = frame.shape[0]

                if canvas is None or canvas.shape != frame.shape:
                    canvas = np.zeros_like(frame)
                    previous_point = None

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = next_timestamp(last_timestamp_ms)
                last_timestamp_ms = timestamp_ms

                result = detector.detect_for_video(mp_image, timestamp_ms)
                status, previous_point = process_hand(
                    frame, canvas, result, previous_point
                )

                output = cv2.add(frame, canvas)
                draw_status(output, status, height)

                cv2.imshow("Air Drawing", output)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    canvas.fill(0)
                    previous_point = None

                elif key in (ord("q"), 27):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()