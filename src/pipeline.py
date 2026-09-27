from pathlib import Path
import time
from hazard_detector import detect_hazards
import numpy as np
import torch
from PIL import Image

from recognition import recognize
from contracts import FrameInput, PipelineResult, FaceResult, Detection
from scene_detector import detect_scene


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_WIDTH = 1600
INPUT_HEIGHT = 1200

MODEL_SIZE = 160
EMBEDDING_DIMENSIONS = 512
FACE_MARGIN = 0.20


# ============================================================
# 1. CREATE DUMMY IMAGE
# ============================================================

def create_dummy_image():
    """
    Create a fake camera image with 1600x1200 resolution.
    """

    image_array = np.zeros(
        (INPUT_HEIGHT, INPUT_WIDTH, 3),
        dtype=np.uint8
    )

    # Simple gray background
    image_array[:, :] = [120, 120, 120]

    image = Image.fromarray(image_array, "RGB")

    return image


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Basic image preprocessing.
    """

    # Ensure RGB
    image = image.convert("RGB")

    # Resize for the model
    image = image.resize((MODEL_SIZE, MODEL_SIZE))

    # Convert to NumPy
    image_array = np.asarray(image, dtype=np.float32)

    # Normalize to 0-1
    image_array = image_array / 255.0

    # Convert HWC -> CHW
    image_array = np.transpose(image_array, (2, 0, 1))

    # Add batch dimension
    tensor = torch.from_numpy(image_array).unsqueeze(0)

    return tensor


# ============================================================
# 3. DUMMY FACE DETECTOR
# ============================================================

def dummy_face_detector(image):
    """
    Pretend that a face detector found a face.

    Returns:
        x1, y1, x2, y2, confidence
    """

    width, height = image.size

    x1 = int(width * 0.30)
    y1 = int(height * 0.20)

    x2 = int(width * 0.70)
    y2 = int(height * 0.80)

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": 0.95,
    }


# ============================================================
# 4. FACE CROP
# ============================================================

def crop_face(image, bbox):
    """
    Crop the detected face with a configurable margin.
    """

    x1 = bbox["x1"]
    y1 = bbox["y1"]
    x2 = bbox["x2"]
    y2 = bbox["y2"]

    width = x2 - x1
    height = y2 - y1

    margin_x = int(width * FACE_MARGIN)
    margin_y = int(height * FACE_MARGIN)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)

    x2 = min(image.width, x2 + margin_x)
    y2 = min(image.height, y2 + margin_y)

    return image.crop((x1, y1, x2, y2))


# ============================================================
# 5. DUMMY EMBEDDING MODEL
# ============================================================

def create_dummy_embedding(face_tensor, device):
    """
    Create a fake 512-dimensional embedding.

    Later this function will be replaced with
    a real face-recognition model.
    """

    embedding = torch.randn(
        1,
        EMBEDDING_DIMENSIONS,
        device=device
    )

    # Normalize embedding
    embedding = embedding / embedding.norm(
        dim=1,
        keepdim=True
    )

    return embedding


# ============================================================
# 6. PIPELINE INTERFACE
# ============================================================

def run_pipeline(frame: FrameInput) -> PipelineResult:
    """
    Run the current dummy face pipeline on one input frame.

    This is the future integration point for real models.
    """

    start_time = time.perf_counter()

    result = PipelineResult(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
    )

    try:
        # Select CPU now, GPU automatically later
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Ensure RGB
        image = frame.image.convert("RGB")
        # ----------------------------------------------------
        # Scene / object detection
        # ----------------------------------------------------

        scene = detect_scene(
            image.width,
            image.height
        )
        hazards = detect_hazards(scene)
        result.hazards.extend(hazards)

        result.depth_available = scene.depth_available

        for obj in scene.objects:
            result.detections.append(
                Detection(
                    label=obj.label,
                    confidence=obj.confidence,
                    x1=obj.bbox.x1,
                    y1=obj.bbox.y1,
                    x2=obj.bbox.x2,
                    y2=obj.bbox.y2,
                )
            )

        # ----------------------------------------------------
        # Dummy face detection
        # ----------------------------------------------------

        bbox = dummy_face_detector(image)

        # ----------------------------------------------------
        # Face crop
        # ----------------------------------------------------

        face = crop_face(image, bbox)

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        face_tensor = preprocess_image(face).to(device)

        # ----------------------------------------------------
        # Dummy embedding
        # ----------------------------------------------------

        embedding = create_dummy_embedding(
            face_tensor,
            device
        )

        # ----------------------------------------------------
        # Dummy recognition gallery
        # ----------------------------------------------------

        gallery = {
            "Person_A": create_dummy_embedding(
                face_tensor,
                device
            ),
            "Person_B": create_dummy_embedding(
                face_tensor,
                device
            ),
            "Person_C": create_dummy_embedding(
                face_tensor,
                device
            ),
        }

        identity, similarity = recognize(
            embedding,
            gallery,
            threshold=0.5
        )

        # ----------------------------------------------------
        # Store face result
        # ----------------------------------------------------

        result.faces.append(
            FaceResult(
                identity=identity,
                confidence=bbox["confidence"],
                similarity=similarity,
                x1=bbox["x1"],
                y1=bbox["y1"],
                x2=bbox["x2"],
                y2=bbox["y2"],
            )
        )

    except Exception as exc:
        result.errors.append(str(exc))

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    result.processing_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return result


# ============================================================
# 7. ORIGINAL DUMMY PIPELINE DEMO
# ============================================================

def main():

    print("=" * 50)
    print("        THIRD EYE DUMMY PIPELINE")
    print("=" * 50)

    start_time = time.perf_counter()

    # Select CPU now, GPU automatically later
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:")
    print(f"  {device}")

    # --------------------------------------------------------
    # Create dummy camera image
    # --------------------------------------------------------

    image = create_dummy_image()

    print("\nInput:")
    print(f"  Resolution: {image.width} x {image.height}")
    print("  RGB: OK")

    # --------------------------------------------------------
    # Dummy detection
    # --------------------------------------------------------

    bbox = dummy_face_detector(image)

    print("\nDetection:")
    print("  Face: detected")
    print(f"  Confidence: {bbox['confidence']:.2f}")

    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    face = crop_face(image, bbox)

    print("\nFace crop:")
    print(f"  Resolution: {face.width} x {face.height}")

    # Save face crop
    output_path = Path("outputs/results/dummy_face.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    face.save(output_path)

    print(f"  Saved: {output_path}")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    face_tensor = preprocess_image(face)

    face_tensor = face_tensor.to(device)

    print("\nPreprocessing:")
    print(f"  Model resolution: {MODEL_SIZE} x {MODEL_SIZE}")
    print(f"  Tensor shape: {tuple(face_tensor.shape)}")
    print(f"  Tensor device: {face_tensor.device}")

    # --------------------------------------------------------
    # Dummy embedding
    # --------------------------------------------------------

    embedding = create_dummy_embedding(
        face_tensor,
        device
    )

    print("\nEmbedding:")
    print(f"  Dimensions: {embedding.shape[1]}")
    print(f"  Device: {embedding.device}")

    # --------------------------------------------------------
    # Dummy recognition gallery
    # --------------------------------------------------------

    gallery = {
        "Person_A": create_dummy_embedding(
            face_tensor,
            device
        ),
        "Person_B": create_dummy_embedding(
            face_tensor,
            device
        ),
        "Person_C": create_dummy_embedding(
            face_tensor,
            device
        ),
    }

    identity, similarity = recognize(
        embedding,
        gallery,
        threshold=0.5
    )

    print("\nRecognition:")
    print(f"  Identity: {identity}")
    print(f"  Similarity: {similarity:.4f}")

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    print("\nPerformance:")
    print(f"  Total pipeline time: {elapsed * 1000:.2f} ms")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\nPipeline: SUCCESS")

    print("=" * 50)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()