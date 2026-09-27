from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class FrameInput:
    frame_id: int
    timestamp: float
    image: Any
    width: int
    height: int
    source: str = "camera"


@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class FaceResult:
    identity: str
    confidence: float
    similarity: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Hazard:
    type: str
    confidence: float
    distance: Optional[float] = None


@dataclass
class PipelineResult:
    frame_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    faces: List[FaceResult] = field(default_factory=list)
    hazards: List[Hazard] = field(default_factory=list)
    depth_available: bool = False
    instruction: str = "INSUFFICIENT_CONFIDENCE"
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
