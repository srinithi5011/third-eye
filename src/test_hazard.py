from scene_detector import detect_scene
from hazard_detector import detect_hazards


def main():
    scene = detect_scene(1600, 1200)
    hazards = detect_hazards(scene)

    print("=" * 50)
    print("THIRD EYE HAZARD TEST")
    print("=" * 50)

    print("\nHazards:")

    for hazard in hazards:
        print(
            f"  {hazard.type}: "
            f"confidence={hazard.confidence:.2f}, "
            f"distance={hazard.distance}"
        )

    print("\nPipeline: SUCCESS")
    print("=" * 50)


if __name__ == "__main__":
    main()