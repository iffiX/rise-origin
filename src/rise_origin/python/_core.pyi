# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)

from typing import Any, Callable, Iterable, List, Optional, Union, overload
import numpy as np

RS_NULL_INDEX: int
RS_WORLD_MESSAGE_SYSTEM_ID: int
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
    source_controller_name: str


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
    has_entity_info: bool


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
]

_RSC_SignalAlt = Union[RSC_Force, RSC_Torque]

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
    def __init__(self, configs: Iterable[str], throw_error: bool = True) -> None:
        """Construct from an iterable of XML configuration strings."""
        ...

    name: str
    is_valid: bool
    invalid_reason: str
    simulation: RSC_Simulation
    resources: VecResourceVariant
    essences: VecEssenceVariant
    entities: VecEntityVariant
    records: VecRecordVariant
    signals: VecSignalVariant
    recorders: VecRecorderVariant
    controllers: VecControllerVariant

    def open(self, paths: Iterable[str], throw_error: bool = True) -> None: ...

    def validate(self, throw_error: bool = True) -> None: ...


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
    RSE_WORLD_MESSAGE_ANY_LEVEL: int
    RSE_WORLD_MESSAGE_INFO: int
    RSE_WORLD_MESSAGE_WARNING: int
    RSE_WORLD_MESSAGE_ERROR: int

class RSE_WorldMessageCreatedType:
    RSE_WORLD_MESSAGE_CREATED_TYPE_NONE: int
    RSE_WORLD_MESSAGE_CREATED_TYPE_ENTITY: int
    RSE_WORLD_MESSAGE_CREATED_TYPE_RESOURCE: int
    RSE_WORLD_MESSAGE_CREATED_TYPE_ESSENCE: int
    RSE_WORLD_MESSAGE_CREATED_TYPE_RECORD: int
    RSE_WORLD_MESSAGE_CREATED_TYPE_SIGNAL: int

class RS_WorldMessageText:
    text: str

    def __init__(self) -> None: ...

class RS_WorldMessageCreated:
    type: RSE_WorldMessageCreatedType
    gid: int
    name: str

    def __init__(self) -> None: ...

class RS_WorldMessageData:
    def __init__(self) -> None: ...

    @property
    def value(self) -> Optional[Union[RS_WorldMessageText, RS_WorldMessageCreated]]: ...

class RS_WorldMessage:
    id: int
    level: RSE_WorldMessageLevel
    data: RS_WorldMessageData

    def __init__(self) -> None: ...


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

        def pixels(self) -> np.ndarray:
            """[height, width, 3] uint8 RGB array."""
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


class RS_WorldIntersectRegionRecord:
    """Region-membership record snapshot. Lists the structures whose voxels lie
    inside the region this record step via the nested ``Membership`` accessor."""

    class Membership:
        def size(self) -> int: ...

        def structure_gid(self) -> np.ndarray: ...

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


################################################################################
# World Controllers
################################################################################

# A record held by a controller. One of the world record snapshot types.
_RS_WorldRecord = Union[
    RS_WorldStructureVoxelRecord,
    RS_WorldCameraPerspectiveRecord,
    RS_WorldIntersectRegionRecord,
]

# A signal held by a controller. One of the world signal snapshot types.
_RS_WorldSignal = Union[
    RS_WorldStructureVoxelSignal,
    RS_WorldCameraPerspectiveSignal,
    RS_WorldForceSignal,
    RS_WorldTorqueSignal,
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

    def is_pending(self) -> bool: ...

    def is_running(self) -> bool: ...

    def is_paused(self) -> bool: ...

    def is_end(self) -> bool: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def terminate(self) -> None: ...

    def get_status(self) -> int: ...

    def wait_for_running(self) -> bool: ...

    def wait_for_end(self) -> None: ...

    def wait_for_message(
        self,
        message_id: int = ...,
        level: RSE_WorldMessageLevel = ...,
    ) -> RS_WorldMessage: ...

    def begin_modification_transaction(self) -> None: ...

    def end_modification_transaction(self) -> None: ...

    def get_world(self) -> RSC_World: ...

    def set_simulation(self, simulation: RSC_Simulation) -> int: ...

    def add_resource(self, resource: RSC_ResourceVariant) -> int: ...

    def remove_resource(self, name: str) -> int: ...

    def find_resource(self, name: str) -> RSC_ResourceVariant: ...

    def add_essence(self, essence: RSC_EssenceVariant) -> int: ...

    def update_essence(self, name: str, essence: RSC_EssenceVariant) -> int: ...

    def remove_essence(self, name: str) -> int: ...

    def find_essence(self, name: str) -> RSC_EssenceVariant: ...

    def add_entity(self, entity: RSC_EntityVariant) -> int: ...

    def remove_entity(self, name: str) -> int: ...

    def freeze_entity(self, name: str) -> int: ...

    def unfreeze_entity(self, name: str) -> int: ...

    def find_entity(self, name: str) -> RSC_EntityVariant: ...

    def is_entity_frozen(self, name: str) -> bool: ...

    def add_record(self, record: RSC_RecordVariant) -> int: ...

    def remove_record(self, name: str) -> int: ...

    def find_record(self, name: str) -> RSC_RecordVariant: ...

    def add_signal(self, signal: RSC_SignalVariant) -> int: ...

    def remove_signal(self, name: str) -> int: ...

    def find_signal(self, name: str) -> RSC_SignalVariant: ...

    def add_recorder(self, recorder: RSC_RecorderVariant) -> int: ...

    def remove_recorder(self, name: str) -> int: ...

    def find_recorder(self, name: str) -> RSC_RecorderVariant: ...

    def add_controller(self, controller: RSC_ControllerVariant) -> int: ...

    def remove_controller(self, name: str) -> int: ...

    def find_controller(self, name: str) -> RSC_ControllerVariant: ...


################################################################################
################################################################################
# init.cpp - module entry point
################################################################################
################################################################################


################################################################################
# Main Rise Class
################################################################################

class RiseOrigin:
    """Top-level entry point for running a world."""

    @overload
    def __init__(self, device: RS_DeviceLocation = ...) -> None: ...

    @overload
    def __init__(self, device_str: str = "") -> None: ...

    def hello(
            self,
            world: RSC_World,
            callback: Optional[
                Callable[[str, float, List[RS_WorldController]], None]
            ] = None,
    ) -> RS_WorldHandle: ...
