"""Show a webcam feed beside a character matching the user's expression."""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer


WINDOW_NAME = "Webcam Expression Character"
EXPRESSIONS = ("neutral", "happy", "surprised", "sad")
CONFIDENCE_THRESHOLD = 0.35
SAD_CONFIDENCE_THRESHOLD = 0.18
SAD_NEUTRAL_MARGIN = 0.18
FACE_PADDING = 0.18
SWITCH_DELAY_SECONDS = 0.30
INFERENCE_INTERVAL_SECONDS = 0.12

MODEL_SCORE_INDEX = {
    "happy": 4,
    "neutral": 5,
    "sad": 6,
    "surprised": 7,
}


def load_characters(base_dir: Path) -> dict[str, np.ndarray]:
    """Load each required drawing once, failing with a useful path."""
    characters = {}
    for expression in EXPRESSIONS:
        path = base_dir / "pics" / f"{expression}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing character image: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read character image: {path}")
        characters[expression] = image
    return characters


def largest_face(
    detector: cv2.CascadeClassifier, frame: np.ndarray
) -> tuple[int, int, int, int] | None:
    """Return the largest detected face, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    return int(x), int(y), int(width), int(height)


def padded_face(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a face with context for eyebrows, cheeks, and jawline."""
    x, y, width, height = box
    padding_x = round(width * FACE_PADDING)
    padding_y = round(height * FACE_PADDING)
    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(frame.shape[1], x + width + padding_x)
    bottom = min(frame.shape[0], y + height + padding_y)
    return frame[top:bottom, left:right]


def expression_from_scores(scores: np.ndarray) -> tuple[str, float]:
    """Choose among the four drawing states, with a gentler sadness rule."""
    target_scores = {
        expression: float(scores[index])
        for expression, index in MODEL_SCORE_INDEX.items()
    }
    expression = max(target_scores, key=target_scores.get)

    sad_score = target_scores["sad"]
    neutral_score = target_scores["neutral"]
    if (
        expression in ("neutral", "sad")
        and sad_score >= SAD_CONFIDENCE_THRESHOLD
        and sad_score >= neutral_score - SAD_NEUTRAL_MARGIN
    ):
        return "sad", sad_score

    confidence = target_scores[expression]
    if expression not in ("neutral", "sad") and confidence < CONFIDENCE_THRESHOLD:
        return "neutral", confidence
    return expression, confidence


def predict_scores(
    recognizer: HSEmotionRecognizer, face_image: np.ndarray
) -> np.ndarray:
    """Return the model's eight expression probabilities."""
    _, scores = recognizer.predict_emotions(face_image, logits=False)
    return np.asarray(scores, dtype=np.float32)


def update_stable_expression(
    current: str, pending: str | None, pending_since: float, predicted: str, now: float
) -> tuple[str, str | None, float]:
    """Switch only after a prediction remains unchanged for a short time."""
    if predicted == current:
        return current, None, 0.0
    if predicted != pending:
        return current, predicted, now
    if now - pending_since >= SWITCH_DELAY_SECONDS:
        return predicted, None, 0.0
    return current, pending, pending_since


def fit_with_padding(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Fit an image into a panel without changing its aspect ratio."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    x = (width - size[0]) // 2
    y = (height - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized
    return canvas


def create_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face detector."""
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {path}")
    return detector


def run() -> None:
    base_dir = Path(__file__).resolve().parent
    characters = load_characters(base_dir)
    detector = create_detector()
    print("Loading expression model (the first run may download its weights)...")
    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")

    camera = cv2.VideoCapture(0)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                "Could not open the default webcam. Check macOS System Settings > "
                "Privacy & Security > Camera and allow access for Terminal/Codex."
            )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        current = "neutral"
        pending: str | None = None
        pending_since = 0.0
        last_inference = 0.0
        face_box = None
        confidence = 0.0

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            frame = cv2.flip(frame, 1)
            now = time.monotonic()

            if now - last_inference >= INFERENCE_INTERVAL_SECONDS:
                face_box = largest_face(detector, frame)
                if face_box is None:
                    current, pending, pending_since = "neutral", None, 0.0
                    confidence = 0.0
                else:
                    scores = predict_scores(recognizer, padded_face(frame, face_box))
                    predicted, confidence = expression_from_scores(scores)
                    current, pending, pending_since = update_stable_expression(
                        current, pending, pending_since, predicted, now
                    )
                last_inference = now

            if face_box is not None:
                x, y, width, height = face_box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 210, 70), 2)

            status = current if face_box is not None else "neutral (no face)"
            cv2.putText(
                frame,
                f"Expression: {status}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if face_box is not None:
                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.0%}",
                    (18, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            panel = fit_with_padding(characters[current], frame.shape[0], frame.shape[0])
            cv2.imshow(WINDOW_NAME, np.hstack((frame, panel)))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> int:
    try:
        run()
    except (FileNotFoundError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
