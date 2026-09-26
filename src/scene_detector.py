from scene import (
    BoundingBox,
    DetectedObject,
    DetectedFace,
    SceneState,
)


def detect_scene(width: int, height: int) -> SceneState:
    """
    Dummy scene detector.

    Later this will be replaced by real
    object, person, and face detection models.
    """

    scene = SceneState(
        image_width=width,
        image_height=height,
    )

    # Dummy person
    scene.objects.append(
        DetectedObject(
            label="person",
            confidence=0.95,
            bbox=BoundingBox(
                x1=450,
                y1=250,
                x2=750,
                y2=900,
            ),
        )
    )

    # Dummy car
    scene.objects.append(
        DetectedObject(
            label="car",
            confidence=0.91,
            bbox=BoundingBox(
                x1=950,
                y1=500,
                x2=1350,
                y2=850,
            ),
        )
    )

    # Dummy road
    scene.objects.append(
        DetectedObject(
            label="road",
            confidence=0.98,
            bbox=BoundingBox(
                x1=0,
                y1=700,
                x2=1600,
                y2=1200,
            ),
        )
    )

    # Dummy face
    scene.faces.append(
        DetectedFace(
            confidence=0.94,
            bbox=BoundingBox(
                x1=520,
                y1=280,
                x2=680,
                y2=480,
            ),
            identity="Unknown",
            similarity=0.32,
        )
    )

    # Pretend depth is available
    scene.depth_available = True

    return scene