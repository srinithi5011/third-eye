from scene import SceneState


def get_object_position(obj, scene):
    """
    Determine whether an object is on the left,
    center, or right side of the scene.
    """

    center_x = (obj.bbox.x1 + obj.bbox.x2) / 2
    scene_center = scene.image_width / 2

    left_boundary = scene.image_width * 0.33
    right_boundary = scene.image_width * 0.66

    if center_x < left_boundary:
        return "left"

    if center_x > right_boundary:
        return "right"

    return "center"


def make_instruction(scene: SceneState) -> str:
    """
    Dummy decision engine.

    Uses object presence and spatial position.
    Later this will also use tracking, depth,
    temporal evidence, and hazard confidence.
    """

    persons = [
        obj for obj in scene.objects
        if obj.label == "person"
    ]

    cars = [
        obj for obj in scene.objects
        if obj.label == "car"
    ]

    # Person and vehicle both present.
    if persons and cars:
        return "Wait"

    # Person detected.
    if persons:
        position = get_object_position(persons[0], scene)

        if position == "left":
            return "Person on left"

        if position == "right":
            return "Person on right"

        return "Person ahead"

    return "Scene clear"