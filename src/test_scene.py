from scene_detector import detect_scene
from decision import make_instruction


WIDTH = 1600
HEIGHT = 1200


def main():

    print("=" * 50)
    print("        THIRD EYE SCENE PIPELINE")
    print("=" * 50)

    scene = detect_scene(WIDTH, HEIGHT)

    print("\nScene:")
    print(f"  Resolution: {scene.image_width} x {scene.image_height}")

    print("\nObjects:")

    for obj in scene.objects:
        print(
            f"  {obj.label}: "
            f"confidence={obj.confidence:.2f}"
        )

    print("\nFaces:")

    for face in scene.faces:
        print(
            f"  {face.identity}: "
            f"confidence={face.confidence:.2f}, "
            f"similarity={face.similarity:.2f}"
        )

    print("\nDepth:")
    print(f"  Available: {scene.depth_available}")

    instruction = make_instruction(scene)

    scene.instruction = instruction

    print("\nDecision:")
    print(f"  Instruction: {scene.instruction}")

    print("\nSpeaker:")
    print(f'  🔊 "{scene.instruction}"')

    print("\nPipeline: SUCCESS")

    print("=" * 50)


if __name__ == "__main__":
    main()