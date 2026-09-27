import time

from contracts import FrameInput
from pipeline import create_dummy_image, run_pipeline


def main():
    image = create_dummy_image()

    frame = FrameInput(
        frame_id=1,
        timestamp=time.time(),
        image=image,
        width=image.width,
        height=image.height,
        source="dummy_camera",
    )

    result = run_pipeline(frame)

    print("=" * 50)
    print("THIRD EYE UNIFIED PIPELINE TEST")
    print("=" * 50)

    print("\nFrame:")
    print(f"  ID: {result.frame_id}")
    print(f"  Resolution: {image.width} x {image.height}")

    print("\nObjects:")
    for detection in result.detections:
        print(
            f"  {detection.label}: "
            f"confidence={detection.confidence:.2f} "
            f"bbox=({detection.x1},{detection.y1},"
            f"{detection.x2},{detection.y2})"
        )
    print("\nHazards:")
    for hazard in result.hazards:
      print(
        f"  {hazard.type}: "
        f"confidence={hazard.confidence:.2f}, "
        f"distance={hazard.distance}"
    )
    print("\nFaces:")
    for face in result.faces:
        print(
            f"  {face.identity}: "
            f"confidence={face.confidence:.2f}, "
            f"similarity={face.similarity:.4f}"
        )

    print("\nDepth:")
    print(f"  Available: {result.depth_available}")

    print("\nPerformance:")
    print(f"  Processing time: {result.processing_time_ms:.2f} ms")

    print("\nErrors:")
    print(f"  {result.errors}")

    print("\nPipeline:")
    if not result.errors:
        print("  SUCCESS")
    else:
        print("  FAILED")

    print("=" * 50)


if __name__ == "__main__":
    main()