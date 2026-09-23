"""
BAS Autonomous HAR System - Core Data Types & Data Contracts
Defines standardized data classes, enums, and structures shared across all 8 agents.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import time


class EntityState(str, Enum):
    DOCKED = "DOCKED"
    APPROACHED = "APPROACHED"
    GRASPED = "GRASPED"
    EXTRACTED = "EXTRACTED"
    RELEASED = "RELEASED"


class HOIAction(str, Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    CONTACT = "CONTACT"
    GRASP = "GRASP"
    EXTRACT = "EXTRACT"
    RELEASE = "RELEASE"


class FSMStep(int, Enum):
    IDLE = 0
    BOX_OPENED = 1
    OBJECT_EXTRACTED = 2
    OBJECT_RETURNED = 3
    COMPLETE = 4
    BOX_CLOSED = 5

    # Semantic Aliases for Dual-Box Procedure
    CONTAINER_OPEN = 1
    RED_EXTRACTED = 2
    YELLOW_EXTRACTED = 3
    OBJECTS_RETURNED = 4
    DUAL_COMPLETE = 5

    def get_name(self, is_dual: bool = False) -> str:
        if is_dual:
            mapping = {
                0: "IDLE",
                1: "CONTAINER_OPEN",
                2: "RED_EXTRACTED",
                3: "YELLOW_EXTRACTED",
                4: "OBJECTS_RETURNED",
                5: "BOX_CLOSED"
            }
            return mapping.get(self.value, self.name)
        return self.name


class AnomalyType(str, Enum):
    NONE = "NONE"
    ERROR_SEQ = "ERROR_SEQ"
    ERROR_SKIP = "ERROR_SKIP"
    ERROR_REGRESSION = "ERROR_REGRESSION"
    ERROR_TECHNIQUE = "ERROR_TECHNIQUE"
    STALL_TIMEOUT = "STALL_TIMEOUT"


# Event-to-condition mapping for config-driven forbidden_events enforcement
FORBIDDEN_EVENT_CONDITIONS = {
    "YELLOW_BOX_EXTRACTED": lambda objects, lid_angle, hoi: (
        "yellow_box" in objects
        and (not objects["yellow_box"].is_inside_container
             or objects["yellow_box"].state == EntityState.EXTRACTED)
    ),
    "YELLOW_BOX_TOUCHED": lambda objects, lid_angle, hoi: (
        any(h.object_name == "yellow_box"
            and h.action in (HOIAction.CONTACT, HOIAction.GRASP, HOIAction.EXTRACT)
            for h in hoi)
    ),
    "RED_BOX_EXTRACTED": lambda objects, lid_angle, hoi: (
        "red_box" in objects
        and (not objects["red_box"].is_inside_container
             or objects["red_box"].state == EntityState.EXTRACTED)
    ),
    "RED_BOX_TOUCHED": lambda objects, lid_angle, hoi: (
        any(h.object_name == "red_box"
            and h.action in (HOIAction.CONTACT, HOIAction.GRASP, HOIAction.EXTRACT)
            for h in hoi)
    ),
    "LID_CLOSED": lambda objects, lid_angle, hoi: lid_angle <= 12.0,
    "BOX_CLOSED": lambda objects, lid_angle, hoi: lid_angle <= 12.0,
    "CONTAINER_ABANDONED": lambda objects, lid_angle, hoi: (
        not any(h.action != HOIAction.IDLE for h in hoi)
        and lid_angle >= 15.0
    ),
}


@dataclass
class Vector3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def distance_to(self, other: 'Vector3D') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5


@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class BBox2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    def iou(self, other: 'BBox2D') -> float:
        ix1 = max(self.xmin, other.xmin)
        iy1 = max(self.ymin, other.ymin)
        ix2 = min(self.xmax, other.xmax)
        iy2 = min(self.ymax, other.ymax)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (self.width * self.height) + (other.width * other.height) - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Joint3D:
    name: str
    pos_camera: Vector3D = field(default_factory=Vector3D)
    pos_rack: Vector3D = field(default_factory=Vector3D)
    confidence: float = 0.0


@dataclass
class AstronautPose3D:
    joints: Dict[str, Joint3D] = field(default_factory=dict)
    elbow_angle_deg: float = 0.0
    shoulder_angle_deg: float = 0.0
    body_orientation_deg: float = 0.0
    rom_limits_violated: bool = False
    keypoints_2d: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)



@dataclass
class IMUReading:
    node_id: str  # "upper_arm", "forearm", "wrist"
    timestamp: float
    acc: Vector3D
    gyro: Vector3D
    is_stationary: bool = False


@dataclass
class Relation:
    subject_name: str
    object_name: str
    predicate: str
    score: float


@dataclass
class ExperimentObject:
    name: str
    class_name: str
    bbox: Optional[BBox2D] = None
    pos_rack: Vector3D = field(default_factory=Vector3D)
    state: EntityState = EntityState.DOCKED
    is_inside_container: bool = False
    relations: List[Relation] = field(default_factory=list)


@dataclass
class HOIInteraction:
    object_name: str
    hand_name: str
    distance_m: float
    action: HOIAction
    duration_frames: int = 0


@dataclass
class TelemetryEvent:
    timestamp: str
    frame_id: int
    step_id: int
    step_name: str
    event: str
    status: str
    instruction: str
    anomaly: str = "NONE"
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SpatialMetrics:
    distance_to_container_m: float = 999.0
    distance_to_components_m: Dict[str, float] = field(default_factory=dict)
    wrist_in_container_2d: bool = False
