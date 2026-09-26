from dataclasses import dataclass, field
from typing import List


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: BoundingBox


@dataclass
class DetectedFace:
    confidence: float
    bbox: BoundingBox
    identity: str = "Unknown"
    similarity: float = 0.0


@dataclass
class SceneState:
    image_width: int
    image_height: int

    objects: List[DetectedObject] = field(default_factory=list)
    faces: List[DetectedFace] = field(default_factory=list)

    depth_available: bool = False

    instruction: str = ""