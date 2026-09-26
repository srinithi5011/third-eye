from scene import SceneState


def make_instruction(scene: SceneState) -> str:
    """
    Dummy decision engine.

    Later this will use real scene understanding,
    object positions, depth, and tracking.
    """

    has_person = any(
        obj.label == "person"
        for obj in scene.objects
    )

    has_car = any(
        obj.label == "car"
        for obj in scene.objects
    )

    if has_person and has_car:
        return "Wait"

    if has_person:
        return "Person ahead"

    return "Scene clear"