# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, overload
import datetime
import numpy as np

RS_NULL_INDEX: int
RS_NULL_BIG_INDEX: int
__version__: str


################################################################################
################################################################################
# init_def.h - basic types, vector bindings, math expressions
################################################################################
################################################################################


################################################################################
# Basic Type Definitions
################################################################################

class RVec3rf:
    """3D float vector"""

    @overload
    def __init__(self) -> None:
        """Initialize to zero vector"""
        ...

    @overload
    def __init__(self, x: float, y: float, z: float) -> None: ...

    @staticmethod
    def zero() -> "RVec3rf": ...

    @staticmethod
    def one() -> "RVec3rf": ...

    @staticmethod
    def full(value: float) -> "RVec3rf": ...

    x: float
    y: float
    z: float


class RQuat3rf:
    """3D float quaternion"""

    @overload
    def __init__(self) -> None:
        """Initialize to identity quaternion"""
        ...

    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...

    @staticmethod
    def unit() -> "RQuat3rf": ...

    x: float
    y: float
    z: float
    w: float


class RMat3rf:
    """3x3 float matrix (row-major)"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
            self,
            m00: float, m01: float, m02: float,
            m10: float, m11: float, m12: float,
            m20: float, m21: float, m22: float,
    ) -> None: ...

    @staticmethod
    def zero() -> "RMat3rf": ...

    @staticmethod
    def eye() -> "RMat3rf": ...

    @staticmethod
    def diagonal(m00: float, m11: float, m22: float) -> "RMat3rf": ...

    def __getitem__(self, idx: tuple[int, int]) -> float: ...

    def __setitem__(self, idx: tuple[int, int], value: float) -> None: ...

    def to_list(self) -> list[list[float]]: ...

    def from_list(self, rows: list[list[float]]) -> None: ...


class RColor3b:
    """Byte RGB color"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, r: int, g: int, b: int) -> None: ...

    @staticmethod
    def zero() -> "RColor3b": ...

    r: int
    g: int
    b: int


class RColor3f:
    """Float RGB color"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, r: float, g: float, b: float) -> None: ...

    @staticmethod
    def zero() -> "RColor3f": ...

    r: float
    g: float
    b: float


class RBound3rf:
    """3D float bounding box"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
            self,
            min_x: float, min_y: float, min_z: float,
            max_x: float, max_y: float, max_z: float,
    ) -> None: ...

    @staticmethod
    def zero() -> "RBound3rf": ...

    @staticmethod
    def lowest() -> "RBound3rf": ...

    @staticmethod
    def highest() -> "RBound3rf": ...

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float


class RBound3rui:
    """3D unsigned int bounding box"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
            self,
            min_x: int, min_y: int, min_z: int,
            max_x: int, max_y: int, max_z: int,
    ) -> None: ...

    @staticmethod
    def zero() -> "RBound3rui": ...

    @staticmethod
    def lowest() -> "RBound3rui": ...

    @staticmethod
    def highest() -> "RBound3rui": ...

    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int


class RBound3ri:
    """3D signed int bounding box"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
            self,
            min_x: int, min_y: int, min_z: int,
            max_x: int, max_y: int, max_z: int,
    ) -> None: ...

    @staticmethod
    def zero() -> "RBound3ri": ...

    @staticmethod
    def lowest() -> "RBound3ri": ...

    @staticmethod
    def highest() -> "RBound3ri": ...

    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int


################################################################################
# Primitive Vector Bindings
################################################################################

class VecInt16(List[int]): ...


class VecUInt16(List[int]): ...


class VecInt32(List[int]): ...


class VecUInt32(List[int]): ...


class VecInt64(List[int]): ...


class VecUInt64(List[int]): ...


class VecChar(List[int]): ...


class VecUChar(List[int]): ...


class VecBool(List[bool]): ...


class VecFloat(List[float]): ...


class VecDouble(List[float]): ...


class VecString(List[str]): ...


class VecMathExprToken(List["RS_MathExprToken"]): ...


################################################################################
# Float Type Enum
################################################################################

class RSE_FloatType:
    RSE_FLOAT_32: int
    RSE_FLOAT_64: int


################################################################################
# Math Expression System
################################################################################

class RSE_MathExprOperator:
    RSE_ME_END: int
    RSE_ME_CONST: int
    RSE_ME_E: int
    RSE_ME_PI: int
    RSE_ME_VAR: int
    RSE_ME_ADD: int
    RSE_ME_SUB: int
    RSE_ME_MUL: int
    RSE_ME_DIV: int
    RSE_ME_MOD: int
    RSE_ME_POW: int
    RSE_ME_SQRT: int
    RSE_ME_SIN: int
    RSE_ME_COS: int
    RSE_ME_TAN: int
    RSE_ME_ATAN: int
    RSE_ME_LOG: int
    RSE_ME_INT: int
    RSE_ME_ABS: int
    RSE_ME_GREATER_THAN: int
    RSE_ME_LESS_THAN: int
    RSE_ME_GREATER_OR_EQUAL: int
    RSE_ME_LESS_OR_EQUAL: int
    RSE_ME_NOT_EQUAL: int
    RSE_ME_EQUAL: int
    RSE_ME_AND: int
    RSE_ME_OR: int


class RS_MathExprToken:
    """Token in a mathematical expression"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, op: int) -> None: ...

    @overload
    def __init__(self, op: int, val: Any) -> None:
        """val is float for meCONST, List[str] for meVAR."""
        ...

    op: int
    value: Any


class RS_MathExpr:
    """Mathematical expression for simulation conditions"""

    def __init__(self) -> None: ...

    def empty(self) -> bool: ...

    def __getitem__(self, idx: int) -> RS_MathExprToken: ...

    def __setitem__(self, idx: int, token: RS_MathExprToken) -> None: ...

    expr: List["RS_MathExprToken"]

    def __str__(self) -> str: ...

    def from_string(self, expr_string: str) -> "RS_MathExpr": ...


################################################################################
################################################################################
# init_rsc.h - RSC characterization tree (essences, resources, entities, signals,
#              controllers, recorders, variants, world)
################################################################################
################################################################################

################################################################################
# RSC Exceptions
################################################################################

class RS_DataException(Exception):
    """Raised when characterization data cannot be parsed."""


class RS_InvalidCharacterizationException(RS_DataException):
    """Raised when a characterization fails validation."""


################################################################################
# RSC Bases
################################################################################

class RSC_Base: ...


class RSC_ResourceBase(RSC_Base):
    name: str


class RSC_EssenceBase(RSC_Base):
    name: str


class RSC_EntityBase(RSC_Base):
    name: str


class RSC_RecordBase(RSC_Base):
    name: str
    target_controller_reference: str
    target_recorder_reference: str


class RSC_SignalBase(RSC_Base):
    name: str
    source_controller_reference: str


class RSC_RecorderBase(RSC_Base):
    name: str


class RSC_ControllerBase(RSC_Base):
    name: str


################################################################################
# Simulation
################################################################################

class RSC_SimulationEnvironment(RSC_Base):
    def __init__(self) -> None: ...

    gravity_acceleration: float


class RSC_SimulationTime(RSC_Base):
    def __init__(self) -> None: ...

    dt: Optional[float]
    dt_ratio: float
    dt_update_interval: int
    dt_softbody_and_rigidbody_update: Optional[float]
    dt_rigidbody_contact_and_constraint_update: Optional[float]
    dt_contact_update: Optional[float]
    dt_mechanical_signal_update: Optional[float]
    dt_sentinel_update: Optional[float]
    is_aligned_to_wallclock: bool


class RSC_SimulationSoftBody(RSC_Base):
    def __init__(self) -> None: ...

    internal_damping_z: float
    contact_damping_z: float
    global_damping_z: float
    integration_frequency_multiplier: float
    link_max_compression: float


class RSC_SimulationRigidBody(RSC_Base):
    def __init__(self) -> None: ...

    constraint_iterations: int
    constraint_position_correction_ratio: float
    constraint_warm_start_damping_z: float
    constraint_stiffness_initial: float
    constraint_stiffness_min_threshold: float
    constraint_stiffness_max_threshold: float
    constraint_stiffness_auto_adjust_watt_threshold: float
    constraint_stiffness_auto_adjust_step: float
    contact_iterations: int
    contact_position_correction_iterations: int
    contact_position_correction_ratio: float
    contact_warm_start_damping_z: float
    contact_slop: float
    linear_damping_z: float
    angular_damping_z: float


class RSC_SimulationContact(RSC_Base):
    def __init__(self) -> None: ...

    bvh_rebuild_sub_interval: int
    bvh_refit_sub_interval: int


class RSC_SimulationVisionTrace(RSC_Base):
    def __init__(self) -> None: ...

    firefly_cap: float
    russian_roulette_start_depth: int


class RSC_SimulationVisionReSTIR(RSC_Base):
    def __init__(self) -> None: ...

    is_enabled: bool
    disocclusion_position_threshold: float
    disocclusion_normal_threshold: float
    max_specular_chain: int
    di_m_cap: float
    di_spatial_neighbors: int
    di_spatial_radius: float
    is_gi_bounce_nee_enabled: bool
    gi_m_cap: float
    gi_spatial_neighbors: int
    gi_spatial_radius: float
    gi_validation_period: int


class RSC_SimulationVisionDenoiseNone(RSC_Base):
    def __init__(self) -> None: ...


class RSC_SimulationVisionDenoiseAtrous(RSC_Base):
    def __init__(self) -> None: ...

    atrous_iterations: int
    atrous_min_passes: int
    sigma_color: float
    sigma_normal: float
    sigma_depth: float


class RSC_SimulationVisionDenoiseSVGF(RSC_Base):
    def __init__(self) -> None: ...

    atrous_iterations: int
    atrous_min_passes: int
    sigma_color: float
    sigma_normal: float
    sigma_depth: float
    temporal_alpha: float
    temporal_alpha_moments: float
    temporal_min_history: float


class RSC_SimulationVision(RSC_Base):
    def __init__(self) -> None: ...

    trace: RSC_SimulationVisionTrace
    restir: RSC_SimulationVisionReSTIR
    denoise: Union[
        RSC_SimulationVisionDenoiseNone,
        RSC_SimulationVisionDenoiseAtrous,
        RSC_SimulationVisionDenoiseSVGF,
    ]


class RSC_SimulationSentinel(RSC_Base):
    def __init__(self) -> None: ...

    is_enabled: bool
    degenerate_min_dt: float


class RSC_SimulationControl(RSC_Base):
    def __init__(self) -> None: ...

    frequency: float


class RSC_SimulationRecord(RSC_Base):
    def __init__(self) -> None: ...

    frequency: float


class RSC_SimulationRuntime(RSC_Base):
    def __init__(self) -> None: ...

    max_time: float
    record_buffer_size: int
    is_save_record: bool
    compute_accuracy: int
    memory_accuracy: int
    log_level: str


class RSC_Simulation(RSC_Base):
    def __init__(self) -> None: ...

    seed: int
    stop_condition: RS_MathExpr
    environment: RSC_SimulationEnvironment
    time: RSC_SimulationTime
    soft_body: RSC_SimulationSoftBody
    rigid_body: RSC_SimulationRigidBody
    contact: RSC_SimulationContact
    vision: RSC_SimulationVision
    sentinel: RSC_SimulationSentinel
    control: RSC_SimulationControl
    record: RSC_SimulationRecord
    runtime: RSC_SimulationRuntime


################################################################################
# Resources
################################################################################

class RSC_TextureBase(RSC_ResourceBase): ...


class RSC_TextureConstant(RSC_TextureBase):
    def __init__(self) -> None: ...

    value: RColor3f


class RSC_TextureGrid(RSC_TextureBase):
    def __init__(self) -> None: ...

    fill_color: RColor3f
    line_color: RColor3f
    is_meters: bool
    x_spacing: float
    y_spacing: float
    line_width: float


################################################################################
# Essences
################################################################################

class RSC_Material(RSC_EssenceBase):
    def __init__(self) -> None: ...


class RSC_MaterialUnaryProperty(RSC_EssenceBase):
    material_reference: str


class RSC_MaterialPhysicalPropertyBase(RSC_MaterialUnaryProperty): ...


class RSC_MaterialPhysicalPropertyFixed(RSC_MaterialPhysicalPropertyBase):
    def __init__(self) -> None: ...


class RSC_MaterialPhysicalPropertySoft(RSC_MaterialPhysicalPropertyBase):
    def __init__(self) -> None: ...

    elastic_modulus: float
    fail_stress: float
    density: float
    poissons_ratio: float
    min_expansion: float
    max_expansion: float


class RSC_MaterialPhysicalPropertyRigid(RSC_MaterialPhysicalPropertyBase):
    def __init__(self) -> None: ...

    density: float


class RSC_MaterialPhysicalPropertyAtlas(RSC_MaterialPhysicalPropertyBase):
    def __init__(self) -> None: ...

    mass: float
    com_position: RVec3rf
    inertia: RMat3rf


class RSC_MaterialOpticalPropertyBase(RSC_MaterialUnaryProperty): ...


class RSC_MaterialOpticalPropertySimpleMicrofacet(RSC_MaterialOpticalPropertyBase):
    def __init__(self) -> None: ...

    albedo_texture_reference: str
    roughness_metalness_texture_reference: str
    is_transparent: bool
    ior: float
    emission_texture_reference: str
    emission_intensity: float


class RSC_MaterialPairPropertyBase(RSC_EssenceBase):
    first_material_reference: str
    second_material_reference: str


class RSC_MaterialPairPropertyCoulombFriction(RSC_MaterialPairPropertyBase):
    def __init__(self) -> None: ...

    u_static: float
    u_dynamic: float


class RSC_MaterialPairPropertyRestitution(RSC_MaterialPairPropertyBase):
    def __init__(self) -> None: ...

    restitution: float


################################################################################
# Entities - Cosmic Background
################################################################################

class RSC_CosmicBackgroundBase(RSC_EntityBase): ...


class RSC_CosmicBackgroundUniform(RSC_CosmicBackgroundBase):
    def __init__(self) -> None: ...

    texture_reference: str
    intensity: float
    rotation_rad: float


class RSC_CosmicBackgroundGradient(RSC_CosmicBackgroundBase):
    def __init__(self) -> None: ...

    zenith_color: RColor3b
    horizon_color: RColor3b
    sun_direction: RVec3rf
    sun_color: RColor3b
    sun_angular_radius: float


class RSC_CosmicBackgroundHosekWilkie(RSC_CosmicBackgroundBase):
    def __init__(self) -> None: ...

    sun_direction: RVec3rf
    sun_color: RColor3b
    sun_angular_radius: float
    turbidity: float
    ground_color: RColor3b
    intensity: float


################################################################################
# Entities - Floor
################################################################################

class RSC_FloorBase(RSC_EntityBase):
    material_reference: str


class RSC_FloorFlat(RSC_FloorBase):
    def __init__(self) -> None: ...

    height: float
    uv_scale: float


class RSC_FloorArray(RSC_FloorBase):
    def __init__(self) -> None: ...

    x_size: float
    y_size: float
    x_values: int
    y_values: int
    is_x_repeat: bool
    is_y_repeat: bool
    height: List[float]


################################################################################
# Entities - Structure (Voxel)
################################################################################

class RSE_StructureExpansionControlMode:
    RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION: int
    RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE: int


class RSE_StructureVoxelConstraintType:
    RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT: int
    RSE_STRUCTURE_VOXEL_FIXED_JOINT: int
    RSE_STRUCTURE_VOXEL_HINGE_JOINT: int
    RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT: int
    RSE_STRUCTURE_VOXEL_NULL_JOINT: int


class RSE_StructureVoxelHingeControlMode:
    RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION: int
    RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION: int
    RSE_STRUCTURE_VOXEL_HINGE_CONTROL_TORQUE: int
    RSE_STRUCTURE_VOXEL_HINGE_CONTROL_VELOCITY: int


class RSE_StructureVoxelPrismaticControlMode:
    RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION: int
    RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION: int
    RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_FORCE: int
    RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_VELOCITY: int


class RSE_StructureVoxelRecordVoxelRange:
    RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE: int
    RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_ALL: int
    RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_RANDOM: int


class RSE_StructureVoxelRecordLinkRange:
    RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE: int
    RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_ALL: int
    RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_RANDOM: int


class RSE_StructureVoxelRecordRigidBodyRange:
    RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE: int
    RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL: int


class RSE_StructureVoxelRecordJointRange:
    RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE: int
    RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_ALL: int


class RSC_StructureVoxelBody(RSC_Base):
    def __init__(self) -> None: ...

    relative_origin_position: RVec3rf
    relative_orientation: RQuat3rf
    x_voxels: int
    y_voxels: int
    z_voxels: int
    material_reference_sid: List[int]
    segment_bid: List[int]
    expansion_signal_sid: List[int]
    expansion_control_mode: List[int]
    link: List[bool]
    voxel_size: float


class VecStructureVoxelBody(List[RSC_StructureVoxelBody]): ...


class RSC_StructureVoxelConstraint(RSC_Base):
    def __init__(self) -> None: ...

    type: int
    is_collision_enabled: bool
    a_body_sid: int
    a_segment_bid: int
    b_body_sid: int
    b_segment_bid: int
    relative_anchor: RVec3rf
    hinge_control_mode: int
    hinge_relative_axis: RVec3rf
    hinge_min: float
    hinge_max: float
    hinge_position_control_kp: float
    hinge_position_control_kd: float
    hinge_max_speed: float
    hinge_max_torque: float
    hinge_signal_sid: int
    prismatic_control_mode: int
    prismatic_relative_axis: RVec3rf
    prismatic_min: float
    prismatic_max: float
    prismatic_position_control_kp: float
    prismatic_position_control_kd: float
    prismatic_max_speed: float
    prismatic_max_force: float
    prismatic_signal_sid: int


class VecStructureVoxelConstraint(List[RSC_StructureVoxelConstraint]): ...


class RSC_StructureVoxelRecord(RSC_RecordBase):
    def __init__(self) -> None: ...

    voxel_record_range: int
    link_record_range: int
    rigid_body_record_range: int
    joint_record_range: int
    voxel_sample_num: int
    link_sample_num: int


class RSC_StructureVoxelSignal(RSC_SignalBase):
    def __init__(self) -> None: ...

    expansion_signal_num: int
    hinge_signal_num: int
    prismatic_signal_num: int


class RSC_StructureBase(RSC_EntityBase):
    origin_position: RVec3rf
    orientation: RQuat3rf
    material_references: List[str]


class RSC_StructureVoxel(RSC_StructureBase):
    def __init__(self) -> None: ...

    is_fixed: bool
    bodies: VecStructureVoxelBody
    constraints: VecStructureVoxelConstraint
    record: RSC_StructureVoxelRecord
    signal: RSC_StructureVoxelSignal


################################################################################
# Entities - Camera
################################################################################

class RSC_CameraPerspectiveRecord(RSC_RecordBase):
    def __init__(self) -> None: ...

    has_rgb: bool
    has_depth: bool
    has_normal: bool
    has_motion: bool
    has_entity_info: bool
    is_depth_hyperbolic: bool
    hyperbolic_near_depth: float
    hyperbolic_far_depth: float


class RSC_CameraPerspectiveSignal(RSC_SignalBase):
    def __init__(self) -> None: ...

    position_signal_num: int
    orientation_signal_num: int
    optics_signal_num: int


class RSC_CameraBase(RSC_EntityBase): ...


class RSC_CameraPerspectiveBase(RSC_CameraBase):
    image_width: int
    image_height: int
    samples_per_pixel: int
    max_depth: int
    field_of_view: float
    defocus_angle: float
    focus_distance: float
    record: RSC_CameraPerspectiveRecord
    signal: RSC_CameraPerspectiveSignal


class RSC_CameraPerspectiveFree(RSC_CameraPerspectiveBase):
    def __init__(self) -> None: ...

    origin_position: RVec3rf
    orientation: RQuat3rf


class RSC_CameraPerspectiveRigidBodyLinked(RSC_CameraPerspectiveBase):
    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int
    relative_anchor: RVec3rf
    relative_orientation: RQuat3rf


class RSC_CameraStereoRecord(RSC_RecordBase):
    def __init__(self) -> None: ...

    has_rgb: bool
    has_depth: bool
    has_normal: bool
    has_motion: bool
    has_entity_info: bool
    is_depth_hyperbolic: bool
    hyperbolic_near_depth: float
    hyperbolic_far_depth: float


class RSC_CameraStereoSignal(RSC_SignalBase):
    def __init__(self) -> None: ...


class RSC_CameraStereoBase(RSC_CameraBase):
    image_width: int
    image_height: int
    samples_per_pixel: int
    max_depth: int
    interpupillary_distance: float
    convergence_distance: float
    angle_left: float
    angle_right: float
    angle_up: float
    angle_down: float
    record: RSC_CameraStereoRecord
    signal: RSC_CameraStereoSignal


class RSC_CameraStereoFree(RSC_CameraStereoBase):
    def __init__(self) -> None: ...

    origin_position: RVec3rf
    orientation: RQuat3rf


class RSC_CameraStereoRigidBodyLinked(RSC_CameraStereoBase):
    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int
    relative_anchor: RVec3rf
    relative_orientation: RQuat3rf


class RSC_RaySensorRecord(RSC_RecordBase):
    def __init__(self) -> None: ...

    has_hit: bool
    has_entity_info: bool
    has_entity_detail: bool
    has_material: bool
    has_normal: bool
    has_velocity: bool


class RSC_RaySensorSignal(RSC_SignalBase):
    def __init__(self) -> None: ...

    position_signal_num: int
    orientation_signal_num: int


class RSC_RaySensorBase(RSC_EntityBase):
    ray_directions: list[RVec3rf]
    max_range: float
    record: RSC_RaySensorRecord
    signal: RSC_RaySensorSignal


class RSC_RaySensorFree(RSC_RaySensorBase):
    def __init__(self) -> None: ...

    origin_position: RVec3rf
    orientation: RQuat3rf


class RSC_RaySensorRigidBodyLinked(RSC_RaySensorBase):
    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int
    relative_anchor: RVec3rf
    relative_orientation: RQuat3rf


################################################################################
# Records
################################################################################

class RSC_IntersectRegionCuboid(RSC_RecordBase):
    def __init__(self) -> None: ...

    position: RVec3rf
    orientation: RQuat3rf
    size: RVec3rf


class RSC_IntersectRegionCuboidRigidBodyLinked(RSC_RecordBase):
    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int
    relative_anchor: RVec3rf
    relative_orientation: RQuat3rf
    size: RVec3rf


class RSC_IntersectRegionSphere(RSC_RecordBase):
    def __init__(self) -> None: ...

    position: RVec3rf
    radius: float


class RSC_IntersectRegionSphereRigidBodyLinked(RSC_RecordBase):
    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int
    relative_anchor: RVec3rf
    radius: float


################################################################################
# Signals
################################################################################

class RSC_Force(RSC_SignalBase):
    def __init__(self) -> None: ...

    structure_name: str
    voxel_range: RBound3ri
    body_sid: int
    segment_bid: int


class RSC_Torque(RSC_SignalBase):
    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int


class RSC_Transform(RSC_SignalBase):
    def __init__(self) -> None: ...

    structure_name: str


################################################################################
# Recorders
################################################################################

class RSC_RecorderHDF5(RSC_RecorderBase):
    def __init__(self) -> None: ...

    path: str


################################################################################
# Controllers
################################################################################

class RSC_ControllerCallback(RSC_ControllerBase):
    def __init__(self) -> None: ...


################################################################################
# Variants
################################################################################

class VecResourceVariant(List["RSC_ResourceVariant"]): ...


class VecEssenceVariant(List["RSC_EssenceVariant"]): ...


class VecEntityVariant(List["RSC_EntityVariant"]): ...


class VecRecordVariant(List["RSC_RecordVariant"]): ...


class VecSignalVariant(List["RSC_SignalVariant"]): ...


class VecRecorderVariant(List["RSC_RecorderVariant"]): ...


class VecControllerVariant(List["RSC_ControllerVariant"]): ...


_RSC_ResourceAlt = Union[RSC_TextureConstant, RSC_TextureGrid]

_RSC_EssenceAlt = Union[
    RSC_Material,
    RSC_MaterialPhysicalPropertyFixed,
    RSC_MaterialPhysicalPropertySoft,
    RSC_MaterialPhysicalPropertyRigid,
    RSC_MaterialPhysicalPropertyAtlas,
    RSC_MaterialOpticalPropertySimpleMicrofacet,
    RSC_MaterialPairPropertyCoulombFriction,
    RSC_MaterialPairPropertyRestitution,
]

_RSC_EntityAlt = Union[
    RSC_CosmicBackgroundUniform,
    RSC_CosmicBackgroundGradient,
    RSC_CosmicBackgroundHosekWilkie,
    RSC_FloorFlat,
    RSC_FloorArray,
    RSC_StructureVoxel,
    RSC_CameraPerspectiveFree,
    RSC_CameraPerspectiveRigidBodyLinked,
]

_RSC_RecordAlt = Union[
    RSC_IntersectRegionCuboid,
    RSC_IntersectRegionCuboidRigidBodyLinked,
    RSC_IntersectRegionSphere,
    RSC_IntersectRegionSphereRigidBodyLinked,
]

_RSC_SignalAlt = Union[RSC_Force, RSC_Torque, RSC_Transform]

_RSC_RecorderAlt = RSC_RecorderHDF5

_RSC_ControllerAlt = RSC_ControllerCallback


class RSC_ResourceVariant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, value: _RSC_ResourceAlt) -> None: ...

    value: _RSC_ResourceAlt


class RSC_EssenceVariant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, value: _RSC_EssenceAlt) -> None: ...

    value: _RSC_EssenceAlt


class RSC_EntityVariant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, value: _RSC_EntityAlt) -> None: ...

    value: _RSC_EntityAlt


class RSC_RecordVariant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, value: _RSC_RecordAlt) -> None: ...

    value: _RSC_RecordAlt


class RSC_SignalVariant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, value: _RSC_SignalAlt) -> None: ...

    value: _RSC_SignalAlt


class RSC_RecorderVariant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, value: _RSC_RecorderAlt) -> None: ...

    value: _RSC_RecorderAlt


class RSC_ControllerVariant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, value: _RSC_ControllerAlt) -> None: ...

    value: _RSC_ControllerAlt


################################################################################
# World
################################################################################

class RSC_World(RSC_Base):
    """Top-level RSC world. Holds simulation, entities, essences, resources,
    controllers, and recorders."""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, rsc_strings: Iterable[str]) -> None:
        """Construct from an iterable of XML characterization strings.

        Raises:
            RS_DataException: When an input string is not valid XML.
        """
        ...

    name: str
    simulation: RSC_Simulation
    resources: VecResourceVariant
    essences: VecEssenceVariant
    entities: VecEntityVariant
    records: VecRecordVariant
    signals: VecSignalVariant
    recorders: VecRecorderVariant
    controllers: VecControllerVariant

    def open(self, paths: Iterable[str]) -> None:
        """Open and parse XML characterization files.

        Raises:
            RS_DataException: When a path cannot be opened or holds invalid XML.
        """
        ...


################################################################################
# Validator
################################################################################

class RS_RSCValidator:
    """Validates characterizations, deduping shared geometry across calls."""

    def __init__(self) -> None: ...

    def validate(self, rsc: RSC_Base) -> None:
        """Validate a characterization and raise on the first failed check.

        Args:
            rsc: Any RSC_Base subclass, such as an RSC_World, to validate.

        Raises:
            RS_InvalidCharacterizationException: When a check fails.
        """
        ...


################################################################################
################################################################################
# init_core.h - runtime device, world records / signals / controllers / handle
################################################################################
################################################################################


################################################################################
# Device Type / World Status / Controller Type Enums
################################################################################

class RSE_DeviceType:
    RSE_DEVICE_TYPE_UNKNOWN: int
    RSE_DEVICE_TYPE_CPU: int
    RSE_DEVICE_TYPE_CUDA: int


class RSE_WorldStatus:
    RSE_WORLD_END_SUCCESS: int
    RSE_WORLD_END_TERMINATED: int
    RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG: int
    RSE_WORLD_END_FAILED_WITH_SIMULATION_EXCEPTION: int
    RSE_WORLD_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION: int
    RSE_WORLD_UNINITIALIZED: int
    RSE_WORLD_INITIALIZING: int
    RSE_WORLD_RUNNING: int
    RSE_WORLD_PAUSED: int
    RSE_WORLD_TERMINATING: int
    RSE_WORLD_MAX_NORMAL_STATE: int


################################################################################
# Device Location
################################################################################

class RS_DeviceLocation:
    """Device location for Rise simulation"""

    @overload
    def __init__(self) -> None:
        """Initialize with default CPU device"""
        ...

    @overload
    def __init__(self, device_str: str) -> None:
        """Initialize from a string ("cpu" or "cuda:<id>")."""
        ...

    @overload
    def __init__(self, type: int, id: int) -> None: ...

    def type(self) -> int: ...

    def id(self) -> int: ...

    def __str__(self) -> str: ...


################################################################################
# World Messages
################################################################################

class RSE_WorldMessageLevel:
    RSE_WORLD_MESSAGE_INFO: int
    RSE_WORLD_MESSAGE_WARNING: int
    RSE_WORLD_MESSAGE_ERROR: int


class RS_WorldMessage:
    index: int
    time: datetime.datetime
    level: RSE_WorldMessageLevel
    text: str

    def __init__(self) -> None: ...

################################################################################
# World Tasks
################################################################################

class RSE_WorldTaskState:
    RSE_WORLD_TASK_PENDING: int
    RSE_WORLD_TASK_SUCCESS: int
    RSE_WORLD_TASK_FAIL: int


class RS_WorldTaskDataCreated:
    gid: int
    name: str

    def __init__(self) -> None: ...


class RS_WorldTaskDataSuccess:
    def __init__(self) -> None: ...


class RS_WorldTaskDataFailed:
    error: str

    def __init__(self) -> None: ...


class RS_WorldTaskData:
    def __init__(self) -> None: ...

    @property
    def value(
        self,
    ) -> Optional[Union[RS_WorldTaskDataCreated, RS_WorldTaskDataSuccess, RS_WorldTaskDataFailed]]: ...


class RS_WorldTask:
    def __init__(self) -> None: ...

    def state(self) -> RSE_WorldTaskState: ...

    def is_finished(self) -> bool: ...

    def is_valid(self) -> bool: ...

    def wait(self) -> None: ...

    def wait_for(self, timeout_seconds: float) -> RSE_WorldTaskState: ...

    def get(self) -> RS_WorldTaskData: ...


################################################################################
# World Records
################################################################################

class RS_WorldStructureVoxelRecord:
    """Voxel-structure record snapshot. Mirrors the C++ nested layout via
    nested ``Voxel`` / ``Link`` / ``RigidBody`` / ``Joint`` accessor classes."""

    class Voxel:
        def size(self) -> int: ...

        def body_sid(self) -> np.ndarray: ...

        def voxel_sid(self) -> np.ndarray: ...

        def segment_bid(self) -> np.ndarray: ...

        def index_x(self) -> np.ndarray: ...

        def index_y(self) -> np.ndarray: ...

        def index_z(self) -> np.ndarray: ...

        def connectivity(self) -> np.ndarray: ...

        def is_surface(self) -> np.ndarray: ...

        def is_rigid(self) -> np.ndarray: ...

        def material(self) -> np.ndarray: ...

        def expansion(self) -> np.ndarray: ...

        def orientation(self) -> np.ndarray:
            """[N, 4] quaternion components."""
            ...

        def position(self) -> np.ndarray:
            """[N, 3] position components."""
            ...

        def ppp_offset(self) -> np.ndarray: ...

        def nnn_offset(self) -> np.ndarray: ...

        def linear_velocity(self) -> np.ndarray: ...

        def angular_velocity(self) -> np.ndarray: ...

        def poissons_strain(self) -> np.ndarray: ...

    class Link:
        def size(self) -> int: ...

        def voxel_neg_sid(self) -> np.ndarray: ...

        def voxel_pos_sid(self) -> np.ndarray: ...

        def is_failed(self) -> np.ndarray: ...

        def strain(self) -> np.ndarray: ...

        def stress(self) -> np.ndarray: ...

    class RigidBody:
        def size(self) -> int: ...

        def rigid_body_sid(self) -> np.ndarray: ...

        def body_sid(self) -> np.ndarray: ...

        def segment_bid(self) -> np.ndarray: ...

        def mass(self) -> np.ndarray: ...

        def orientation(self) -> np.ndarray:
            """[N, 4] quaternion components."""
            ...

        def com(self) -> np.ndarray:
            """[N, 3] center-of-mass positions."""
            ...

        def linear_velocity(self) -> np.ndarray: ...

        def angular_velocity(self) -> np.ndarray: ...

    class Joint:
        def size(self) -> int: ...

        def rigid_body_a_sid(self) -> np.ndarray: ...

        def rigid_body_b_sid(self) -> np.ndarray: ...

        def type(self) -> np.ndarray: ...

        def position(self) -> np.ndarray:
            """[N, 3] joint anchor positions."""
            ...

        def axis(self) -> np.ndarray:
            """[N, 3] joint axes."""
            ...

        def hinge_signal_sid(self) -> np.ndarray: ...

        def hinge_min(self) -> np.ndarray: ...

        def hinge_max(self) -> np.ndarray: ...

        def hinge_angle(self) -> np.ndarray: ...

        def hinge_torque(self) -> np.ndarray: ...

        def prismatic_min(self) -> np.ndarray: ...

        def prismatic_max(self) -> np.ndarray: ...

        def prismatic_distance(self) -> np.ndarray: ...

        def prismatic_force(self) -> np.ndarray: ...

    def step(self) -> int: ...

    def time(self) -> float: ...

    def structure_gid(self) -> int: ...

    @property
    def voxels(self) -> "RS_WorldStructureVoxelRecord.Voxel": ...

    @property
    def links(self) -> "RS_WorldStructureVoxelRecord.Link": ...

    @property
    def rigid_bodies(self) -> "RS_WorldStructureVoxelRecord.RigidBody": ...

    @property
    def joints(self) -> "RS_WorldStructureVoxelRecord.Joint": ...


class RS_WorldCameraPerspectiveRecord:
    """Perspective-camera record snapshot. Mirrors the C++ nested layout via
    nested ``RGB`` / ``State`` accessor classes."""

    class RGB:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width, 3] uint8 RGB array."""
            ...

    class Depth:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width] float32 depth array, empty when disabled."""
            ...

    class Normal:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width, 3] float32 world-space normal array, empty when disabled."""
            ...

    class Motion:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width, 2] float32 screen-space motion array, empty when disabled."""
            ...

    class EntityInfo:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def entity_type(self) -> np.ndarray:
            """[height, width] uint32 entity type ids, empty when disabled."""
            ...

        def entity_gid(self) -> np.ndarray:
            """[height, width] uint32 entity global ids, empty when disabled."""
            ...

    class State:
        def size(self) -> int: ...

        def position(self) -> np.ndarray:
            """[size, 3] camera positions."""
            ...

        def orientation(self) -> np.ndarray:
            """[size, 4] camera orientations (x, y, z, w)."""
            ...

        def focus_distance(self) -> np.ndarray:
            """[size] focus distances."""
            ...

        def field_of_view(self) -> np.ndarray:
            """[size] vertical fields of view."""
            ...

    def step(self) -> int: ...

    def time(self) -> float: ...

    def camera_gid(self) -> int: ...

    def structure_gid(self) -> int: ...

    def rigid_body_sid(self) -> int: ...

    def width(self) -> int: ...

    def height(self) -> int: ...

    def is_valid(self) -> bool: ...

    @property
    def rgb(self) -> "RS_WorldCameraPerspectiveRecord.RGB": ...

    @property
    def state(self) -> "RS_WorldCameraPerspectiveRecord.State": ...


class RS_WorldCameraStereoRecord:
    """Stereo-camera record snapshot. Each eye is exposed through left_/right_
    accessors mirroring the C++ nested layout."""

    class RGB:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width, 3] uint8 RGB array."""
            ...

    class Depth:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width] float32 depth array, empty when disabled."""
            ...

    class Normal:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width, 3] float32 world-space normal array, empty when disabled."""
            ...

    class Motion:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def pixel(self) -> np.ndarray:
            """[height, width, 2] float32 screen-space motion array, empty when disabled."""
            ...

    class EntityInfo:
        def width(self) -> int: ...

        def height(self) -> int: ...

        def entity_type(self) -> np.ndarray:
            """[height, width] uint32 entity type ids, empty when disabled."""
            ...

        def entity_gid(self) -> np.ndarray:
            """[height, width] uint32 entity global ids, empty when disabled."""
            ...

    class State:
        def size(self) -> int: ...

        def position(self) -> np.ndarray:
            """[size, 3] camera positions."""
            ...

        def orientation(self) -> np.ndarray:
            """[size, 4] camera orientations (x, y, z, w)."""
            ...

        def focus_distance(self) -> np.ndarray:
            """[size] focus distances."""
            ...

        def field_of_view(self) -> np.ndarray:
            """[size] vertical fields of view."""
            ...

    def step(self) -> int: ...

    def time(self) -> float: ...

    def camera_gid(self) -> int: ...

    def structure_gid(self) -> int: ...

    def rigid_body_sid(self) -> int: ...

    def width(self) -> int: ...

    def height(self) -> int: ...

    def is_valid(self) -> bool: ...

    @property
    def left_rgb(self) -> "RS_WorldCameraStereoRecord.RGB": ...

    @property
    def right_rgb(self) -> "RS_WorldCameraStereoRecord.RGB": ...

    @property
    def left_depth(self) -> "RS_WorldCameraStereoRecord.Depth": ...

    @property
    def right_depth(self) -> "RS_WorldCameraStereoRecord.Depth": ...

    @property
    def left_normal(self) -> "RS_WorldCameraStereoRecord.Normal": ...

    @property
    def right_normal(self) -> "RS_WorldCameraStereoRecord.Normal": ...

    @property
    def left_motion(self) -> "RS_WorldCameraStereoRecord.Motion": ...

    @property
    def right_motion(self) -> "RS_WorldCameraStereoRecord.Motion": ...

    @property
    def left_entity_info(self) -> "RS_WorldCameraStereoRecord.EntityInfo": ...

    @property
    def right_entity_info(self) -> "RS_WorldCameraStereoRecord.EntityInfo": ...

    @property
    def left_camera_state(self) -> "RS_WorldCameraStereoRecord.State": ...

    @property
    def right_camera_state(self) -> "RS_WorldCameraStereoRecord.State": ...


class RS_WorldRaySensorRecord:
    """Ray-sensor record snapshot. Per-ray channels are exposed through nested
    accessor classes, each returning [ray_num, ...] arrays that are empty when
    the channel is disabled."""

    class Hit:
        def size(self) -> int: ...

        def is_hit(self) -> np.ndarray:
            """[ray_num] uint8 hit flags."""
            ...

        def distance(self) -> np.ndarray:
            """[ray_num] float32 hit distances."""
            ...

        def hit_position(self) -> np.ndarray:
            """[ray_num, 3] float32 world hit positions."""
            ...

    class EntityInfo:
        def size(self) -> int: ...

        def entity_gid(self) -> np.ndarray:
            """[ray_num] uint32 hit entity global ids."""
            ...

        def entity_type(self) -> np.ndarray:
            """[ray_num] uint32 hit entity type ids."""
            ...

    class EntityDetail:
        def size(self) -> int: ...

        def body_sid(self) -> np.ndarray:
            """[ray_num] uint32 hit body ids."""
            ...

        def segment_bid(self) -> np.ndarray:
            """[ray_num] uint32 hit segment ids."""
            ...

        def voxel_sid(self) -> np.ndarray:
            """[ray_num] uint32 hit voxel ids."""
            ...

    class Material:
        def size(self) -> int: ...

        def material_id(self) -> np.ndarray:
            """[ray_num] uint32 hit material ids."""
            ...

        def emission(self) -> np.ndarray:
            """[ray_num] float32 hit material emission."""
            ...

    class Normal:
        def size(self) -> int: ...

        def normal(self) -> np.ndarray:
            """[ray_num, 3] float32 world hit normals."""
            ...

    class Velocity:
        def size(self) -> int: ...

        def v_hit(self) -> np.ndarray:
            """[ray_num, 3] float32 world velocity of the hit surface."""
            ...

        def v_rel(self) -> np.ndarray:
            """[ray_num, 3] float32 hit velocity relative to the sensor."""
            ...

        def v_radial(self) -> np.ndarray:
            """[ray_num] float32 radial component of the relative velocity."""
            ...

    class State:
        def size(self) -> int: ...

        def position(self) -> np.ndarray:
            """[size, 3] sensor positions."""
            ...

        def orientation(self) -> np.ndarray:
            """[size, 4] sensor orientations (x, y, z, w)."""
            ...

    def step(self) -> int: ...

    def time(self) -> float: ...

    def sensor_gid(self) -> int: ...

    def structure_gid(self) -> int: ...

    def rigid_body_sid(self) -> int: ...

    def ray_num(self) -> int: ...

    def is_valid(self) -> bool: ...

    @property
    def hit(self) -> "RS_WorldRaySensorRecord.Hit": ...

    @property
    def entity_info(self) -> "RS_WorldRaySensorRecord.EntityInfo": ...

    @property
    def entity_detail(self) -> "RS_WorldRaySensorRecord.EntityDetail": ...

    @property
    def material(self) -> "RS_WorldRaySensorRecord.Material": ...

    @property
    def normal(self) -> "RS_WorldRaySensorRecord.Normal": ...

    @property
    def velocity(self) -> "RS_WorldRaySensorRecord.Velocity": ...

    @property
    def state(self) -> "RS_WorldRaySensorRecord.State": ...


class RS_WorldIntersectRegionRecord:
    """Region-membership record snapshot. Lists the structures whose voxels lie
    inside the region this record step via the nested ``Membership`` accessor."""

    class Membership:
        def size(self) -> int: ...

        def structure_gid(self) -> np.ndarray: ...

        def closest_relative_position(self) -> np.ndarray: ...

    def step(self) -> int: ...

    def time(self) -> float: ...

    def record_gid(self) -> int: ...

    @property
    def memberships(self) -> "RS_WorldIntersectRegionRecord.Membership": ...


################################################################################
# World Signals
################################################################################

class RS_WorldStructureVoxelSignal:
    class Expansion:
        def size(self) -> int: ...

        def expansion(self) -> np.ndarray: ...

    class Hinge:
        def size(self) -> int: ...

        def hinge(self) -> np.ndarray: ...

    class Prismatic:
        def size(self) -> int: ...

        def prismatic(self) -> np.ndarray: ...

    def expansion_signals(self) -> RS_WorldStructureVoxelSignal.Expansion: ...

    def hinge_signals(self) -> RS_WorldStructureVoxelSignal.Hinge: ...

    def prismatic_signals(self) -> RS_WorldStructureVoxelSignal.Prismatic: ...


class RS_WorldCameraPerspectiveSignal:
    class Position:
        def size(self) -> int: ...

        def position(self) -> np.ndarray: ...

    class Orientation:
        def size(self) -> int: ...

        def orientation(self) -> np.ndarray: ...

    class Optics:
        def size(self) -> int: ...

        def focus_distance(self) -> np.ndarray: ...

        def field_of_view(self) -> np.ndarray: ...

    def position_signals(self) -> RS_WorldCameraPerspectiveSignal.Position: ...

    def orientation_signals(self) -> RS_WorldCameraPerspectiveSignal.Orientation: ...

    def optics_signals(self) -> RS_WorldCameraPerspectiveSignal.Optics: ...


class RS_WorldCameraStereoSignal:
    class Position:
        def size(self) -> int: ...

        def position(self) -> np.ndarray: ...

    class Orientation:
        def size(self) -> int: ...

        def orientation(self) -> np.ndarray: ...

    class Optics:
        def size(self) -> int: ...

        def field_of_view(self) -> np.ndarray: ...

    def left_position_signals(self) -> RS_WorldCameraStereoSignal.Position: ...

    def right_position_signals(self) -> RS_WorldCameraStereoSignal.Position: ...

    def left_orientation_signals(self) -> RS_WorldCameraStereoSignal.Orientation: ...

    def right_orientation_signals(self) -> RS_WorldCameraStereoSignal.Orientation: ...

    def left_optics_signals(self) -> RS_WorldCameraStereoSignal.Optics: ...

    def right_optics_signals(self) -> RS_WorldCameraStereoSignal.Optics: ...


class RS_WorldRaySensorSignal:
    class Position:
        def size(self) -> int: ...

        def position(self) -> np.ndarray: ...

    class Orientation:
        def size(self) -> int: ...

        def orientation(self) -> np.ndarray: ...

    def position_signals(self) -> RS_WorldRaySensorSignal.Position: ...

    def orientation_signals(self) -> RS_WorldRaySensorSignal.Orientation: ...


class RS_WorldForceSignal:
    class Force:
        def size(self) -> int: ...

        def force(self) -> np.ndarray: ...

    def force_signals(self) -> RS_WorldForceSignal.Force: ...


class RS_WorldTorqueSignal:
    class Torque:
        def size(self) -> int: ...

        def torque(self) -> np.ndarray: ...

    def torque_signals(self) -> RS_WorldTorqueSignal.Torque: ...


class RS_WorldTransformSignal:
    class Translation:
        def size(self) -> int: ...

        def translation(self) -> np.ndarray: ...

    class Rotation:
        def size(self) -> int: ...

        def rotation(self) -> np.ndarray: ...

        def pivot(self) -> np.ndarray: ...

    def translation_signals(self) -> RS_WorldTransformSignal.Translation: ...

    def rotation_signals(self) -> RS_WorldTransformSignal.Rotation: ...


################################################################################
# World Controllers
################################################################################

# A record held by a controller. One of the world record snapshot types.
_RS_WorldRecord = Union[
    RS_WorldStructureVoxelRecord,
    RS_WorldCameraPerspectiveRecord,
    RS_WorldCameraStereoRecord,
    RS_WorldIntersectRegionRecord,
]

# A signal held by a controller. One of the world signal snapshot types.
_RS_WorldSignal = Union[
    RS_WorldStructureVoxelSignal,
    RS_WorldCameraPerspectiveSignal,
    RS_WorldCameraStereoSignal,
    RS_WorldForceSignal,
    RS_WorldTorqueSignal,
    RS_WorldTransformSignal,
]


class RS_WorldController:
    """Generic controller payload delivered to the world callback. Bundles the
    records and signals routed to one controller this callback step."""

    def name(self) -> str: ...

    def records(self) -> List[_RS_WorldRecord]: ...

    def signals(self) -> List[_RS_WorldSignal]: ...


################################################################################
# World Handle
################################################################################

class RS_WorldHandle:
    """Live handle to a running world. Mutations must be framed by
    ``begin_modification_transaction`` / ``end_modification_transaction``."""

    def is_pending(self) -> bool:
        """Report whether the world is still uninitialized or initializing.

        Returns:
            True when the world has not yet started running.
        """
        ...

    def is_running(self) -> bool:
        """Report whether the world is currently running.

        Returns:
            True when the world is in the running state.
        """
        ...

    def is_paused(self) -> bool:
        """Report whether the world is currently paused.

        Returns:
            True when the world is in the paused state.
        """
        ...

    def is_end(self) -> bool:
        """Report whether the world has ended.

        Returns:
            True when the world has reached any end state, whether clean or failed.
        """
        ...

    def pause(self) -> None:
        """Pause the running world.

        Blocks until the world is running, then requests a pause. Does nothing if the
        world ends before it starts running.
        """
        ...

    def resume(self) -> None:
        """Resume the paused world.

        Blocks until the world is running, then requests a resume. Does nothing if the
        world ends before it starts running.
        """
        ...

    def terminate(self) -> None:
        """Terminate the running world.

        Blocks until the world is running, then requests termination. Does nothing if
        the world ends before it starts running.
        """
        ...

    def get_status(self) -> int:
        """Read the current world status.

        Returns:
            The current RSE_WorldStatus.
        """
        ...

    def wait_for_running(self) -> bool:
        """Block until the world starts running or ends.

        Returns:
            True if the world reached the running state, False if it ended first.
        """
        ...

    def wait_for_end(self) -> None:
        """Block until the world ends."""
        ...

    def get_newest_message_index(self) -> int:
        """Index of the newest message in the log, or RS_NULL_BIG_INDEX when empty."""
        ...

    def get_oldest_message_index(self) -> int:
        """Index of the oldest retained message; a cursor below it has missed entries."""
        ...

    def get_messages_matching(
        self,
        max_count: int = ...,
        oldest_index: int = ...,
        newest_index: int = ...,
        level: RSE_WorldMessageLevel = ...,
    ) -> List[RS_WorldMessage]:
        """Return up to max_count messages in [oldest_index, newest_index) at severity >= level.

        Reads never remove. Results run oldest to newest; when capped, the newest are kept.

        Returns:
            The matching RS_WorldMessage objects.
        """
        ...

    def wait_for_message(self, level: RSE_WorldMessageLevel = ...) -> RS_WorldMessage:
        """Block until the next message at severity >= level, then return it."""
        ...

    def get_message_buffer_size(self) -> int:
        """Number of entries the message log retains."""
        ...

    def set_message_buffer_size(self, max_messages: int) -> None:
        """Set the number of entries the message log retains; shrinking evicts the oldest."""
        ...

    def begin_modification_transaction(self) -> None:
        """Open a modification transaction so several edits apply on the same step.

        Blocks until the world is running. Does nothing if the world ends before it
        starts running.
        """
        ...

    def end_modification_transaction(self) -> None:
        """Close the open modification transaction and apply its batched edits.

        Blocks until the world is running. Does nothing if the world ends before it
        starts running.
        """
        ...

    def get_world(self) -> RSC_World:
        """Snapshot the current world characterization.

        Blocks until the world is running, then reads back the live world including its
        active recorders and controllers.

        Returns:
            The current RSC_World, or an empty RSC_World if the world ended first.
        """
        ...

    def set_simulation(self, simulation: RSC_Simulation) -> RS_WorldTask:
        """Replace the simulation parameters of the running world.

        Args:
            simulation: The RSC_Simulation to apply.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def add_resource(self, resource: RSC_ResourceVariant) -> RS_WorldTask:
        """Validate and add a resource to the running world.

        Args:
            resource: The RSC_ResourceVariant to add.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the resource fails validation.
        """
        ...

    def remove_resource(self, name: str) -> RS_WorldTask:
        """Remove the named resource from the running world.

        Args:
            name: The name of the resource to remove.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def find_resource(self, name: str) -> RSC_ResourceVariant:
        """Look up the named resource in the running world.

        Args:
            name: The name of the resource to find.

        Returns:
            The matching RSC_ResourceVariant, or an empty variant if the world is not running.
        """
        ...

    def add_essence(self, essence: RSC_EssenceVariant) -> RS_WorldTask:
        """Validate and add an essence to the running world.

        Args:
            essence: The RSC_EssenceVariant to add.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the essence fails validation.
        """
        ...

    def update_essence(self, name: str, essence: RSC_EssenceVariant) -> RS_WorldTask:
        """Validate and update the named essence in the running world.

        Args:
            name: The name of the essence to update.
            essence: The RSC_EssenceVariant holding the new values.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the essence fails validation.
        """
        ...

    def remove_essence(self, name: str) -> RS_WorldTask:
        """Remove the named essence from the running world.

        Args:
            name: The name of the essence to remove.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def find_essence(self, name: str) -> RSC_EssenceVariant:
        """Look up the named essence in the running world.

        Args:
            name: The name of the essence to find.

        Returns:
            The matching RSC_EssenceVariant, or an empty variant if the world is not running.
        """
        ...

    def add_entity(self, entity: RSC_EntityVariant) -> RS_WorldTask:
        """Validate and add an entity to the running world.

        Args:
            entity: The RSC_EntityVariant to add.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the entity fails validation.
        """
        ...

    def remove_entity(self, name: str) -> RS_WorldTask:
        """Remove the named entity from the running world.

        Args:
            name: The name of the entity to remove.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def freeze_entity(self, name: str) -> RS_WorldTask:
        """Freeze the named entity so it stops advancing.

        Args:
            name: The name of the entity to freeze.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def unfreeze_entity(self, name: str) -> RS_WorldTask:
        """Unfreeze the named entity so it resumes advancing.

        Args:
            name: The name of the entity to unfreeze.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def find_entity(self, name: str) -> RSC_EntityVariant:
        """Look up the named entity in the running world.

        Args:
            name: The name of the entity to find.

        Returns:
            The matching RSC_EntityVariant, or an empty variant if the world is not running.
        """
        ...

    def is_entity_frozen(self, name: str) -> bool:
        """Report whether the named entity is frozen.

        Args:
            name: The name of the entity to query.

        Returns:
            True if the entity is frozen, False if it is not or the world is not running.
        """
        ...

    def add_record(self, record: RSC_RecordVariant) -> RS_WorldTask:
        """Validate and add a record to the running world.

        Args:
            record: The RSC_RecordVariant to add.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the record fails validation.
        """
        ...

    def remove_record(self, name: str) -> RS_WorldTask:
        """Remove the named record from the running world.

        Args:
            name: The name of the record to remove.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def find_record(self, name: str) -> RSC_RecordVariant:
        """Look up the named record in the running world.

        Args:
            name: The name of the record to find.

        Returns:
            The matching RSC_RecordVariant, or an empty variant if the world is not running.
        """
        ...

    def add_signal(self, signal: RSC_SignalVariant) -> RS_WorldTask:
        """Validate and add a signal to the running world.

        Args:
            signal: The RSC_SignalVariant to add.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the signal fails validation.
        """
        ...

    def remove_signal(self, name: str) -> RS_WorldTask:
        """Remove the named signal from the running world.

        Args:
            name: The name of the signal to remove.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def find_signal(self, name: str) -> RSC_SignalVariant:
        """Look up the named signal in the running world.

        Args:
            name: The name of the signal to find.

        Returns:
            The matching RSC_SignalVariant, or an empty variant if the world is not running.
        """
        ...

    def add_recorder(self, recorder: RSC_RecorderVariant) -> RS_WorldTask:
        """Validate and add a recorder to the running world.

        Args:
            recorder: The RSC_RecorderVariant to add.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the recorder fails validation.
        """
        ...

    def remove_recorder(self, name: str) -> RS_WorldTask:
        """Remove the named recorder from the running world.

        Args:
            name: The name of the recorder to remove.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def find_recorder(self, name: str) -> RSC_RecorderVariant:
        """Look up the named recorder in the running world.

        Args:
            name: The name of the recorder to find.

        Returns:
            The matching RSC_RecorderVariant, or an empty variant if the world is not running.
        """
        ...

    def add_controller(self, controller: RSC_ControllerVariant) -> RS_WorldTask:
        """Validate and add a controller to the running world.

        Args:
            controller: The RSC_ControllerVariant to add.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.

        Raises:
            RS_InvalidCharacterizationException: When the controller fails validation.
        """
        ...

    def remove_controller(self, name: str) -> RS_WorldTask:
        """Remove the named controller from the running world.

        Args:
            name: The name of the controller to remove.

        Returns:
            A task tracking the command; already finished and failed if the world is not running.
        """
        ...

    def find_controller(self, name: str) -> RSC_ControllerVariant:
        """Look up the named controller in the running world.

        Args:
            name: The name of the controller to find.

        Returns:
            The matching RSC_ControllerVariant, or an empty variant if the world is not running.
        """
        ...


################################################################################
# World Callback
################################################################################

class RS_WorldCallback:
    """The rebindable per-step world callback the engine drives each control step."""

    def get_world_name(self) -> str: ...

    def set_callback(
        self, callback: Callable[[float, List[RS_WorldController]], None]
    ) -> None: ...

    def reset_callback(self) -> None: ...

    def get_callback(
        self,
    ) -> Optional[Callable[[float, List[RS_WorldController]], None]]: ...


################################################################################
# World Session Components
################################################################################

class RS_WorldSessionComponent:
    """Base class for session components."""


class RS_WorldSessionComponentGeneric(RS_WorldSessionComponent):
    """Runs a user callback over the controllers in its scope each step."""

    def __init__(self) -> None: ...

    def get_handle(self) -> RS_WorldHandle:
        """This component's scoped world handle."""
        ...

    def set_callback(
        self, callback: Callable[[float, List[RS_WorldController]], None]
    ) -> None:
        """Set the per-step callback, invoked as callback(time, controllers)."""
        ...


class RS_WorldSessionComponentCamera(RS_WorldSessionComponent):
    """Orbit-camera component: injects one free perspective camera and drives its pose."""

    def __init__(
        self,
        name: str,
        callback: Optional[Callable[[float], None]] = None,
        position: RVec3rf = (0.0, 0.0, 0.4),
        focus: RVec3rf = (0.0, 0.0, 0.15),
        width: int = 1280,
        height: int = 720,
        samples_per_pixel: int = 4,
        max_depth: int = 3,
        field_of_view: float = 60.0,
        defocus_angle: float = 0.0,
        focus_distance: float = 1.0,
    ) -> None: ...

    def move(
        self,
        orbit_pitch: float,
        orbit_yaw: float,
        pan_x: float,
        pan_y: float,
        zoom: float,
    ) -> None:
        """Accumulate a mouse gesture; applied on the next control step."""
        ...

    def frame(self) -> Optional[np.ndarray]:
        """The latest RGB frame as a (height, width, 3) uint8 array, or None if empty."""
        ...


class RS_WorldSessionComponentVR(RS_WorldSessionComponent):
    """Stereo VR component: injects one stereo camera and presents it to an OpenXR headset.

    Only present when the core is built with RISE_ORIGIN_BUILD_CORE_WITH_VR.
    """

    def __init__(
        self,
        name: str,
        samples_per_pixel: int = 1,
        max_depth: int = 3,
        near_depth: float = 0.05,
        far_depth: float = 100.0,
        move_speed: float = 2.0,
        fly_speed: float = 1.5,
        turn_speed: float = 90.0,
        stick_dead_zone: float = 0.2,
        boost: float = 3.0,
        hud_follow_tau: float = 0.25,
    ) -> None: ...

    def setPushForce(self, push_force: float) -> None:
        """Set the newtons applied along the aim ray at a full trigger pull."""
        ...


class RS_WorldSessionComponentMCP(RS_WorldSessionComponent):
    """MCP server component: hosts a Model Context Protocol server so an AI client can read, change, see, and be embodied in the world.

    Only present when the core is built with RISE_ORIGIN_BUILD_CORE_WITH_MCP.
    """

    def __init__(
        self,
        name: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str = "",
        is_python_exec_enabled: bool = False,
    ) -> None: ...

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: str,
        handler: Callable[[RS_WorldHandle, str], str],
    ) -> None:
        """Register a Python authored tool served next to the built in ones."""
        ...

    def unregister_tool(self, name: str) -> None:
        """Remove a registered tool by name."""
        ...

    def list_registered_tools(self) -> List[str]:
        """Get the names of every registered tool."""
        ...


################################################################################
# World Session
################################################################################

class RSE_WorldSessionScope:
    RSE_WORLD_SESSION_SCOPE_OWNED: int
    RSE_WORLD_SESSION_SCOPE_FULL: int

    
class RS_WorldSession:
    """Owns a running world's handle and callback and drives its components."""

    def __init__(self, handle: RS_WorldHandle, callback: RS_WorldCallback) -> None: ...

    def has_component(self, name: str) -> bool: ...

    def add_component(
        self,
        name: str,
        component: RS_WorldSessionComponent,
        scope: RSE_WorldSessionScope = ...,
    ) -> None: ...

    def remove_component(self, name: str) -> None: ...

    def get_core_component(self) -> RS_WorldSessionComponentGeneric: ...

    def get_handle(self) -> RS_WorldHandle: ...

    def run(self) -> None: ...


################################################################################
################################################################################
# init.cpp - module entry point
################################################################################
################################################################################


################################################################################
# Main Rise Class
################################################################################

class RiseOrigin:
    """The RiseOrigin.

    Usage::

        world = RSC_World(["config.xml"])
        r = RiseOrigin()
        handle = r.hello(world)
        # World is yours
    """

    @overload
    def __init__(self, device: RS_DeviceLocation = ...) -> None:
        """Construct a local RiseOrigin instance on the specified device.

        Args:
            device: Device identifiers used by this process.
        """
        ...

    @overload
    def __init__(self, device_str: str = "") -> None:
        """Construct a local RiseOrigin instance from a device string.

        Args:
            device_str: Device string, such as "cpu" or "cuda:0". Defaults to the local
                CPU when empty.
        """
        ...

    def hello(self, world: RSC_World) -> Tuple[RS_WorldHandle, RS_WorldCallback]:
        """Say hello to the world.

        Args:
            world: Root world config describing all entities, essences, resources, etc
                that should exist before time starts.

        Returns:
            The world handle and the rebindable world callback the engine drives each step.
        """
        ...

    def connect(self, url: str) -> Tuple[RS_WorldHandle, RS_WorldCallback]:
        """Connect to a world running on an already-launched remote instance.

        Args:
            url: Address of the remote instance hosting the world.

        Returns:
            The world handle and the rebindable world callback the engine drives each step.
        """
        ...
