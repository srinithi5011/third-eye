from scene import SceneState
from contracts import Hazard


def detect_hazards(scene: SceneState):
    """
    Dummy hazard detector.

    Later this will combine:
    - object detection
    - depth
    - tracking
    - spatial relationships
    - temporal evidence
    """

    hazards = []

    has_person = any(
        obj.label == "person"
        for obj in scene.objects
    )

    has_car = any(
        obj.label == "car"
        for obj in scene.objects
    )

    if has_person and has_car:
        hazards.append(
            Hazard(
                type="person_vehicle_interaction",
                confidence=0.90,
                distance=None,
            )
        )

    elif has_person:
        hazards.append(
            Hazard(
                type="person",
                confidence=0.85,
                distance=None,
            )
        )

    return hazards