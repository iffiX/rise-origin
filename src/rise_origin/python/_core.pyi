# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, overload
import numpy as np

# Constants
RS_NULL_INDEX: int

# Version information
__version__: str


################################################################################
# Basic Type Definitions
################################################################################

# 3D Vector
class RVec3rf:
    """3D float vector"""

    @overload
    def __init__(self) -> None:
        """Initialize to zero vector"""
        ...

    @overload
    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Initialize with explicit components.

        Args:
            x: X component
            y: Y component
            z: Z component
        """
        ...

    x: float
    y: float
    z: float


# Quaternion
class RQuat3rf:
    """3D float quaternion"""

    @overload
    def __init__(self) -> None:
        """Initialize to identity quaternion"""
        ...

    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        """
        Initialize with explicit components.

        Args:
            x: X component
            y: Y component
            z: Z component
            w: W component
        """
        ...

    x: float
    y: float
    z: float
    w: float


class RColor3b:
    """8bit RGB color"""

    @overload
    def __init__(self) -> None:
        """Initialize as zero color"""

    @overload
    def __init__(self, r: int, g: int, b: int) -> None:
        """
        Initialize with explicit arguments

        Args:
            r: R component
            g: G component
            b: B component
        """
        ...

    r: int
    g: int
    b: int


class RBound3rf:
    """3D float bounding box"""

    @overload
    def __init__(self) -> None:
        """Initialize to zero bounding box"""
        ...

    @overload
    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float) -> None:
        """
        Initialize with explicit arguments"""
        ...

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float


class RBound3rui:
    """3D unsigned int bounding box"""

    @overload
    def __init__(self) -> None:
        """Initialize to zero bounding box"""
        ...

    @overload
    def __init__(self, min_x: int, max_x: int, min_y: int, max_y: int, min_z: int, max_z: int) -> None:
        """
        Initialize with explicit arguments"""
        ...

    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_z: int
    max_z: int


class RBound3ri:
    """3D signed int bounding box"""

    @overload
    def __init__(self) -> None:
        """Initialize to zero bounding box"""
        ...

    @overload
    def __init__(self, min_x: int, max_x: int, min_y: int, max_y: int, min_z: int, max_z: int) -> None:
        """
        Initialize with explicit arguments"""
        ...

    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_z: int
    max_z: int


################################################################################
# Enumerations
################################################################################

class RSE_FloatType:
    RSE_FLOAT_32: int
    RSE_FLOAT_64: int
    
# Device type enumeration
class RSE_DeviceType:
    RSE_DEVICE_TYPE_UNKNOWN: int
    RSE_DEVICE_TYPE_CPU: int
    RSE_DEVICE_TYPE_CUDA: int


# Simulation status enumeration
class RSE_SimulationStatus:
    RSE_SIMULATION_END_SUCCESS: int
    RSE_SIMULATION_END_TERMINATED: int
    RSE_SIMULATION_END_FAILED_WITH_INVALID_CONFIG: int
    RSE_SIMULATION_END_FAILED_WITH_SIMULATION_EXCEPTION: int
    RSE_SIMULATION_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION: int
    RSE_SIMULATION_UNINITIALIZED: int
    RSE_SIMULATION_INITIALIZING: int
    RSE_SIMULATION_RUNNING: int
    RSE_SIMULATION_PAUSED: int
    RSE_SIMULATION_TERMINATING: int
    RSE_SIMULATION_MAX_NORMAL_STATE: int


# Controller type enumeration
class RSE_SimulationControllerType:
    RSE_SIMULATION_STRUCTURE_CONTROLLER: int
    RSE_SIMULATION_CAMERA_CONTROLLER: int
    RSE_SIMULATION_FORCE_CONTROLLER: int
    RSE_SIMULATION_TORQUE_CONTROLLER: int


# Math expression operator enumeration
class RS_MathExprOperator:
    meEND: int
    meCONST: int
    meE: int
    mePI: int
    meVAR: int
    meADD: int
    meSUB: int
    meMUL: int
    meDIV: int
    meMOD: int
    mePOW: int
    meSQRT: int
    meSIN: int
    meCOS: int
    meTAN: int
    meATAN: int
    meLOG: int
    meINT: int
    meABS: int
    meGREATERTHAN: int
    meLESSTHAN: int
    meGREATEROREQUAL: int
    meLESSOREQUAL: int
    meNOTEQUAL: int
    meEQUAL: int
    meAND: int
    meOR: int


# Floor data type enumeration
class RSE_FloorDataType:
    RSE_FLOOR_DATA_NONE: int
    RSE_FLOOR_DATA_FLAT: int
    RSE_FLOOR_DATA_ARRAY: int


# Floor render type enumeration
class RSE_FloorRenderType:
    RSE_FLOOR_RENDER_OPAQUE: int
    RSE_FLOOR_RENDER_OPAQUE_GRID: int
    RSE_FLOOR_RENDER_TRANSPARENT: int


# World render type enumeration
class RSE_WorldRenderType:
    RSE_WORLD_RENDER_CONSTANT_COLOR: int
    RSE_WORLD_RENDER_GRADIENT: int
    RSE_WORLD_RENDER_HOSEK_WILKIE: int


# Material render type enumeration
class RSE_MaterialRenderType:
    RSE_MATERIAL_RENDER_OPAQUE: int
    RSE_MATERIAL_RENDER_TRANSPARENT: int
    RSE_MATERIAL_RENDER_EMISSIVE: int


# Vision denoise type enumeration
class RSE_VisionDenoiseType:
    RSE_VISION_DENOISE_NONE: int
    RSE_VISION_DENOISE_ATROUS: int
    RSE_VISION_DENOISE_SVGF: int


# Camera type enumeration
class RSE_CameraType:
    RSE_CAMERA_NONE: int
    RSE_CAMERA_FREE_PERSPECTIVE: int
    RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE: int


# Structure constraint type enumeration
class RSE_StructureConstraintType:
    RSE_BALL_AND_SOCKET_JOINT: int
    RSE_FIXED_JOINT: int
    RSE_HINGE_JOINT: int
    RSE_PRISMATIC_JOINT: int
    RSE_NULL_JOINT: int


# Structure hinge joint control mode enumeration
class RSE_StructureHingeJointControlMode:
    RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION: int
    RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION: int
    RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_TORQUE: int
    RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_VELOCITY: int


# Structure prismatic joint control mode enumeration
class RSE_StructurePrismaticJointControlMode:
    RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION: int
    RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION: int
    RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_FORCE: int
    RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_VELOCITY: int


# Record type enumerations
class RSE_StructureVoxelRecordRange:
    RSE_VOXEL_RECORD_RANGE_NONE: int
    RSE_VOXEL_RECORD_RANGE_ALL: int
    RSE_VOXEL_RECORD_RANGE_RANDOM: int


class RSE_StructureLinkRecordRange:
    RSE_LINK_RECORD_RANGE_NONE: int
    RSE_LINK_RECORD_RANGE_ALL: int
    RSE_LINK_RECORD_RANGE_RANDOM: int


class RSE_StructureRigidBodyRecordRange:
    RSE_RIGID_BODY_RECORD_RANGE_NONE: int
    RSE_RIGID_BODY_RECORD_RANGE_ALL: int


class RSE_StructureJointRecordRange:
    RSE_JOINT_RECORD_RANGE_NONE: int
    RSE_JOINT_RECORD_RANGE_ALL: int


class RSE_RecordType:
    RSE_RECORD_STRUCTURE: int


class RSE_SignalType:
    RSE_SIGNAL_FORCE: int
    RSE_SIGNAL_TORQUE: int


class RSE_RecorderType:
    RSE_RECORDER_HDF5: int


################################################################################
# Math Expression System
################################################################################

# Math expression token
class RS_MathExprToken:
    """Token in a mathematical expression"""

    @overload
    def __init__(self) -> None:
        """Initialize empty token"""
        ...

    @overload
    def __init__(self, op: int) -> None:
        """
        Initialize with an operator.

        Args:
            op: Operator from RS_MathExprOperator
        """
        ...

    @overload
    def __init__(self, op: int, val: Any) -> None:
        """
        Initialize with an operator and value.

        Args:
            op: Operator from RS_MathExprOperator
            val: Value - float for meCONST, List[str] for meVAR
        """
        ...

    op: int
    value: Any  # Either float for meCONST or List[str] for meVAR


# Math expression
class RS_MathExpr:
    """Mathematical expression for simulation conditions"""

    def __init__(self) -> None:
        """Initialize empty expression"""
        ...

    def empty(self) -> bool:
        """Check if the expression is empty"""
        ...

    def __getitem__(self, idx: int) -> RS_MathExprToken:
        """Get token at the given index"""
        ...

    def __setitem__(self, idx: int, token: RS_MathExprToken) -> None:
        """Set token at the given index"""
        ...

    expr: List["RS_MathExprToken"]

    def __str__(self) -> str:
        """String representation"""
        ...

    def from_string(self, expr_string: str) -> "RS_MathExpr":
        """
        Parse from string representation.

        Args:
            expr_string: String representation of the expression

        Returns:
            Reference to this expression
        """
        ...


################################################################################
# Floor Configuration
################################################################################

class RS_ArrayFloorDataConfig:
    """Configuration for array-based floor data"""

    def __init__(self) -> None: ...

    x_size: float
    y_size: float
    x_values: int
    y_values: int
    is_x_repeat: bool
    is_y_repeat: bool
    h_min: float
    h_max: float
    height: List[int]


class RS_OpaqueFloorRenderConfig:
    """Configuration for opaque floor rendering"""

    def __init__(self) -> None: ...

    color: RColor3b
    roughness: float
    metalness: float


class RS_OpaqueGridFloorRenderConfig:
    """Configuration for opaque grid floor rendering"""

    def __init__(self) -> None: ...

    x_spacing: float
    y_spacing: float
    line_width: float
    fill_color: RColor3b
    line_color: RColor3b
    roughness: float
    metalness: float


class RS_TransparentFloorRenderConfig:
    """Configuration for transparent floor rendering (tint + IOR)"""

    def __init__(self) -> None: ...

    color: RColor3b
    ior: float
    roughness: float


class RS_FloorDataConfig:
    """Configuration for floor data"""

    def __init__(self) -> None: ...

    type: int
    config: Any  # Can be RS_ArrayFloorDataConfig


class RS_FloorRenderConfig:
    """Configuration for floor rendering"""

    def __init__(self) -> None: ...

    type: int
    config: Any  # Can be RS_OpaqueFloorRenderConfig, RS_OpaqueGridFloorRenderConfig, or RS_TransparentFloorRenderConfig


class RS_FloorConfig:
    """Complete floor configuration"""

    def __init__(self) -> None: ...

    data_config: RS_FloorDataConfig
    render_config: RS_FloorRenderConfig


################################################################################
# World Configuration
################################################################################

class RS_ConstantColorWorldRenderConfig:
    """Configuration for constant color world rendering"""

    def __init__(self) -> None: ...

    color: RColor3b


class RS_GradientSkyWorldRenderConfig:
    """Configuration for two-tone gradient sky with a simple sun disk"""

    def __init__(self) -> None: ...

    zenith_color: RColor3b
    horizon_color: RColor3b
    sun_direction: RVec3rf
    sun_color: RColor3b
    sun_angular_radius: float


class RS_HosekWilkieSkyWorldRenderConfig:
    """Configuration for Hosek-Wilkie 2012/2013 analytic clear-sky model"""

    def __init__(self) -> None: ...

    sun_direction: RVec3rf
    sun_color: RColor3b
    sun_angular_radius: float
    turbidity: float
    ground_color: RColor3b
    intensity: float


class RS_WorldRenderConfig:
    """Configuration for world rendering"""

    def __init__(self) -> None: ...

    type: int
    config: Any  # Can be RS_ConstantColorWorldRenderConfig, RS_GradientSkyWorldRenderConfig, or RS_HosekWilkieSkyWorldRenderConfig


class RS_WorldConfig:
    """Complete world configuration"""

    def __init__(self) -> None: ...

    grav_acc: float
    render_config: RS_WorldRenderConfig


################################################################################
# Material Configuration
################################################################################

class RS_OpaqueMaterialRenderConfig:
    """Configuration for opaque material rendering"""

    def __init__(self) -> None: ...

    color: RColor3b
    roughness: float
    metalness: float


class RS_TransparentMaterialRenderConfig:
    """Configuration for transparent material rendering (tint + IOR)"""

    def __init__(self) -> None: ...

    color: RColor3b
    ior: float
    roughness: float


class RS_EmissiveMaterialRenderConfig:
    """Configuration for emissive material rendering (color * intensity)"""

    def __init__(self) -> None: ...

    color: RColor3b
    intensity: float


class RS_MaterialRenderConfig:
    """Configuration for material rendering"""

    def __init__(self) -> None: ...

    type: int
    config: Any  # Can be RS_OpaqueMaterialRenderConfig, RS_TransparentMaterialRenderConfig, or RS_EmissiveMaterialRenderConfig


class RS_MechanicalMaterialConfig:
    """Mechanical properties of a material"""

    def __init__(self) -> None: ...

    is_fixed: bool
    is_rigid: bool
    min_expansion: float
    max_expansion: float
    elastic_modulus: float
    fail_stress: float
    density: float
    poissons_ratio: float
    u_static: float
    u_dynamic: float


class RS_MaterialConfig:
    """Complete material configuration"""

    def __init__(self) -> None: ...

    name: str
    render_config: RS_MaterialRenderConfig
    mechanical_config: RS_MechanicalMaterialConfig


################################################################################
# Simulation Configuration
################################################################################

class RS_SimulationConfig:
    """Simulation parameters configuration"""

    def __init__(self) -> None: ...

    seed: int
    runtime_dt_update_interval: int
    runtime_rigidbody_update_interval: int
    runtime_contact_update_interval: int
    runtime_contact_bvh_rebuild_sub_interval: int
    runtime_contact_bvh_refit_sub_interval: int
    runtime_mechanical_signal_update_interval: int
    runtime_divergence_check_interval: int
    runtime_record_buffer_size: int
    runtime_max_time: float
    runtime_save_record: bool
    runtime_simulation_compute_accuracy: RSE_FloatType
    runtime_simulation_memory_accuracy: RSE_FloatType
    runtime_log_level: str
    integration_time_step: float
    integration_dt_frac: float
    softbody_internal_damping_z: float
    softbody_contact_damping_z: float
    softbody_global_damping_z: float
    rigidbody_constraint_iterations: int
    rigidbody_constraint_position_correction_ratio: float
    rigidbody_constraint_warm_start_damping_z: float
    rigidbody_contact_iterations: int
    rigidbody_contact_position_correction_iterations: int
    rigidbody_contact_position_correction_ratio: float
    rigidbody_contact_warm_start_damping_z: float
    rigidbody_contact_restitution: float
    rigidbody_contact_slop: float
    rigidbody_linear_damping_z: float
    rigidbody_angular_damping_z: float
    vision_trace_firefly_cap: float
    vision_trace_russian_roulette_start_depth: int
    vision_is_restir_enabled: bool
    vision_restir_disocclusion_position_threshold: float
    vision_restir_disocclusion_normal_threshold: float
    vision_restir_max_specular_chain: int
    vision_restir_di_m_cap: int
    vision_restir_di_spatial_neighbors: int
    vision_restir_di_spatial_radius: float
    vision_restir_gi_is_bounce_nee_enabled: bool
    vision_restir_gi_m_cap: int
    vision_restir_gi_spatial_neighbors: int
    vision_restir_gi_spatial_radius: float
    vision_restir_gi_validation_period: int
    vision_denoise_type: int
    vision_denoise_atrous_iterations: int
    vision_denoise_atrous_min_passes: int
    vision_denoise_sigma_color: float
    vision_denoise_sigma_normal: float
    vision_denoise_sigma_depth: float
    vision_denoise_temporal_alpha: float
    vision_denoise_temporal_alpha_moments: float
    vision_denoise_temporal_min_history: float
    control_frequency: float
    record_frequency: float
    stop_condition: RS_MathExpr


################################################################################
# Camera Configuration
################################################################################

class RS_PerspectiveCameraConfig:
    """Base configuration for perspective cameras"""

    def __init__(self) -> None: ...

    image_width: int
    image_height: int
    samples_per_pixel: int
    max_depth: int
    field_of_view: float
    defocus_angle: float
    focus_distance: float


class RS_FreePerspectiveCameraConfig(RS_PerspectiveCameraConfig):
    """Configuration for free perspective camera"""

    def __init__(self) -> None: ...

    origin_position: RVec3rf
    orientation: RQuat3rf


class RS_RigidBodyLinkedPerspectiveCameraConfig(RS_PerspectiveCameraConfig):
    """Configuration for rigid body linked perspective camera"""

    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int
    relative_anchor: RVec3rf
    relative_orientation: RQuat3rf


class RS_CameraConfig:
    """Complete camera configuration"""

    def __init__(self) -> None: ...

    name: str
    type: int
    config: Any  # Can be RS_FreePerspectiveCameraConfig or RS_RigidBodyLinkedPerspectiveCameraConfig


################################################################################
# Structure Configuration
################################################################################

class RS_StructureBodyConfig:
    """Configuration for a structure body"""

    def __init__(self) -> None: ...

    body_sid: int
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


class RS_StructureTendonConfig:
    """Configuration for a structure tendon"""

    def __init__(self) -> None: ...

    material_reference_sid: int
    a_body_sid: int
    a_voxel_x: int
    a_voxel_y: int
    a_voxel_z: int
    a_voxel_face: int
    b_body_sid: int
    b_voxel_x: int
    b_voxel_y: int
    b_voxel_z: int
    b_voxel_face: int


class RS_StructureConstraintConfig:
    """Configuration for a structure constraint (joint)"""

    def __init__(self) -> None: ...

    type: int
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


class VecStructureBodyConfig(List[RS_StructureBodyConfig]): ...


class VecStructureTendonConfig(List[RS_StructureTendonConfig]): ...


class VecStructureConstraintConfig(List[RS_StructureConstraintConfig]): ...


class RS_StructureConfig:
    """Complete structure configuration"""

    def __init__(self) -> None: ...

    name: str
    is_fixed: bool
    voxel_size: float
    expansion_signal_num: int
    hinge_signal_num: int
    prismatic_signal_num: int
    origin_position: RVec3rf
    orientation: RQuat3rf
    material_references: List[str]
    bodies: VecStructureBodyConfig
    tendons: VecStructureTendonConfig
    constraints: VecStructureConstraintConfig


################################################################################
# Record Configuration
################################################################################

class RS_StructureRecordConfig:
    """Configuration for structure recording"""

    def __init__(self) -> None: ...

    structure_name: str
    voxel_record_range: int
    link_record_range: int
    rigid_body_record_range: int
    joint_record_range: int
    voxel_sample_num: int
    link_sample_num: int


class RS_RecordConfig:
    """Complete record configuration"""

    def __init__(self) -> None: ...

    name: str
    type: int
    config: Any  # Can be RS_StructureRecordConfig


################################################################################
# Signal Configuration
################################################################################

class RS_ForceSignalConfig:
    """Configuration for force signal"""

    def __init__(self) -> None: ...

    structure_name: str
    voxel_range: RBound3ri
    body_sid: int
    segment_bid: int


class RS_TorqueSignalConfig:
    """Configuration for torque signal (rigid bodies only)"""

    def __init__(self) -> None: ...

    structure_name: str
    body_sid: int
    segment_bid: int


class RS_SignalConfig:
    """Complete signal configuration"""

    def __init__(self) -> None: ...

    name: str
    type: int
    config: Any  # Can be RS_ForceSignalConfig or RS_TorqueSignalConfig


class RS_HDF5RecorderConfig:
    """Configuration for HDF5 recorder"""

    def __init__(self) -> None: ...

    path: str


class RS_RecorderConfig:
    """Complete recorder configuration"""

    def __init__(self) -> None: ...

    type: int
    config: Any  # Can be RS_HDF5RecorderConfig


# Vector types for configurations
class VecMaterialConfig(List[RS_MaterialConfig]): ...


class VecStructureConfig(List[RS_StructureConfig]): ...


class VecRecordConfig(List[RS_RecordConfig]): ...


class VecSignalConfig(List[RS_SignalConfig]): ...


class VecRecorderConfig(List[RS_RecorderConfig]): ...


class VecCameraConfig(List[RS_CameraConfig]): ...


################################################################################
# Main Configuration
################################################################################

class RS_Config:
    """Main configuration for RiseOrigin simulation"""

    @overload
    def __init__(self) -> None:
        """Initialize empty configuration"""
        ...

    @overload
    def __init__(self, paths: Iterable[str]) -> None:
        """
        Initialize from configuration files.

        Args:
            paths: Paths to configuration files
        """
        ...

    is_valid: bool
    invalid_reason: str
    simulation_config: RS_SimulationConfig
    floor_config: RS_FloorConfig
    world_config: RS_WorldConfig
    material_configs: VecMaterialConfig
    structure_configs: VecStructureConfig
    record_configs: VecRecordConfig
    signal_configs: VecSignalConfig
    recorder_configs: VecRecorderConfig
    camera_configs: VecCameraConfig

    def open(self, paths: Iterable[str]) -> None:
        """
        Open configuration files.

        Args:
            paths: Paths to configuration files
        """
        ...

    def saveAsBinary(self) -> bytes:
        """
        Serialize the configuration to binary format.

        Returns:
            bytes: Serialized configuration
        """
        ...

    def loadFromBinary(self, data: bytes) -> None:
        """
        Load configuration from binary data.

        Args:
            data: Serialized configuration
        """
        ...


################################################################################
# Device Location
################################################################################

class RS_DeviceLocation:
    """Device location for RiseOrigin simulation"""

    @overload
    def __init__(self) -> None:
        """Initialize with default CPU device"""
        ...

    @overload
    def __init__(self, device_str: str) -> None:
        """
        Initialize from a string representation.

        Args:
            device_str: Either "cpu" or "cuda:<id>"
        """
        ...

    @overload
    def __init__(self, type: int, id: int) -> None:
        """
        Initialize with explicit type and ID.

        Args:
            type: Device type from RSE_DeviceType
            id: Device ID
        """
        ...

    def type(self) -> int:
        """Get the device type"""
        ...

    def id(self) -> int:
        """Get the device ID"""
        ...

    def __str__(self) -> str:
        """String representation (e.g., "cpu" or "cuda:0")"""
        ...


################################################################################
# Simulation Signal
################################################################################

class RS_SimulationStructureSignal:
    """Interface for controlling structures in real-time"""

    def expansion_signals(self) -> Union[np.ndarray, None]:
        """
        Get the expansion signals array.

        Returns:
            numpy.ndarray or None: Memory view of expansion signals if available
        """
        ...

    def hinge_signals(self) -> Union[np.ndarray, None]:
        """
        Get the hinge signals array.

        Returns:
            numpy.ndarray or None: Memory view of hinge signals if available
        """
        ...


class RS_SimulationCameraSignal:
    """Interface for controlling cameras in real-time"""

    def orientation_signals(self) -> Union[np.ndarray, None]:
        """
        Get the rotation velocity signals array.

        Returns:
            numpy.ndarray or None: Memory view of rotation velocity signals if available
        """
        ...

    def position_signals(self) -> Union[np.ndarray, None]:
        """
        Get the translation velocity signals array.

        Returns:
            numpy.ndarray or None: Memory view of translation velocity signals if available
        """
        ...

    def focus_distance_signals(self) -> Union[np.ndarray, None]:
        """
        Get the focus distance signals array.

        Returns:
            numpy.ndarray or None: Memory view of focus distance signals if available
        """
        ...

    def field_of_view_signals(self) -> Union[np.ndarray, None]:
        """
        Get the field of view signals array.

        Returns:
            numpy.ndarray or None: Memory view of field of view signals if available
        """
        ...


class RS_SimulationForceSignal:
    """Interface for controlling force signals in real-time"""

    def force_signals(self) -> Union[np.ndarray, None]:
        """
        Get the force signals array.

        Returns:
            numpy.ndarray or None: Memory view of force signal if available, shape [3] (3D vector)
        """
        ...


class RS_SimulationTorqueSignal:
    """Interface for controlling torque signals in real-time"""

    def torque_signals(self) -> Union[np.ndarray, None]:
        """
        Get the torque signals array.

        Returns:
            numpy.ndarray or None: Memory view of torque signal if available, shape [3] (3D vector)
        """
        ...


################################################################################
# Simulation Record
################################################################################

class RS_SimulationStructureRecord:
    """Interface for accessing structure simulation state"""

    def step(self) -> int:
        """Get the current simulation step"""
        ...

    def time(self) -> float:
        """Get the current simulation time"""
        ...

    def voxel_size(self) -> int:
        """Get the number of voxels"""
        ...

    def link_size(self) -> int:
        """Get the number of links"""
        ...

    def rigid_body_size(self) -> int:
        """Get the number of rigid bodies"""
        ...

    def joint_size(self) -> int:
        """Get the number of joints"""
        ...

    # Voxel data
    def voxel_body_sid(self) -> np.ndarray:
        """Get voxel body structure IDs"""
        ...

    def voxel_voxel_sid(self) -> np.ndarray:
        """Get voxel structure IDs"""
        ...

    def voxel_segment_bid(self) -> np.ndarray:
        """Get voxel segment body IDs"""
        ...

    def voxel_index_x(self) -> np.ndarray:
        """Get voxel X indices"""
        ...

    def voxel_index_y(self) -> np.ndarray:
        """Get voxel Y indices"""
        ...

    def voxel_index_z(self) -> np.ndarray:
        """Get voxel Z indices"""
        ...

    def voxel_connectivity(self) -> np.ndarray:
        """Get voxel connectivity flags"""
        ...

    def voxel_is_surface(self) -> np.ndarray:
        """Get voxel surface flags"""
        ...

    def voxel_is_rigid(self) -> np.ndarray:
        """Get voxel rigidity flags"""
        ...

    def voxel_material(self) -> np.ndarray:
        """Get voxel material indices"""
        ...

    def voxel_expansion(self) -> np.ndarray:
        """Get voxel expansion values"""
        ...

    def voxel_orientation(self) -> np.ndarray:
        """
        Get voxel orientations as quaternions.

        Returns:
            numpy.ndarray: Array of shape [N, 4] with quaternion components
        """
        ...

    def voxel_position(self) -> np.ndarray:
        """
        Get voxel positions.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with position components
        """
        ...

    def voxel_ppp_offset(self) -> np.ndarray:
        """
        Get voxel PPP offsets.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with offset components
        """
        ...

    def voxel_nnn_offset(self) -> np.ndarray:
        """
        Get voxel NNN offsets.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with offset components
        """
        ...

    def voxel_linear_velocity(self) -> np.ndarray:
        """
        Get voxel linear velocities.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with velocity components
        """
        ...

    def voxel_angular_velocity(self) -> np.ndarray:
        """
        Get voxel angular velocities.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with angular velocity components
        """
        ...

    def voxel_poissons_strain(self) -> np.ndarray:
        """
        Get voxel Poisson's strain values.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with strain components
        """
        ...

    # Link data
    def link_voxel_neg_sid(self) -> np.ndarray:
        """Get link negative voxel structure IDs"""
        ...

    def link_voxel_pos_sid(self) -> np.ndarray:
        """Get link positive voxel structure IDs"""
        ...

    def link_is_tendon(self) -> np.ndarray:
        """Get link tendon flags"""
        ...

    def link_is_failed(self) -> np.ndarray:
        """Get link failure flags"""
        ...

    def link_strain(self) -> np.ndarray:
        """Get link strain values"""
        ...

    def link_stress(self) -> np.ndarray:
        """Get link stress values"""
        ...

    # Rigid body data
    def rigid_body_rigid_body_sid(self) -> np.ndarray:
        """Get rigid body structure IDs"""
        ...

    def rigid_body_body_sid(self) -> np.ndarray:
        """Get rigid body parent body structure IDs"""
        ...

    def rigid_body_segment_bid(self) -> np.ndarray:
        """Get rigid body segment body IDs"""
        ...

    def rigid_body_mass(self) -> np.ndarray:
        """Get rigid body masses"""
        ...

    def rigid_body_orientation(self) -> np.ndarray:
        """
        Get rigid body orientations as quaternions.

        Returns:
            numpy.ndarray: Array of shape [N, 4] with quaternion components
        """
        ...

    def rigid_body_com(self) -> np.ndarray:
        """
        Get rigid body centers of mass.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with position components
        """
        ...

    def rigid_body_linear_velocity(self) -> np.ndarray:
        """
        Get rigid body linear velocities.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with velocity components
        """
        ...

    def rigid_body_angular_velocity(self) -> np.ndarray:
        """
        Get rigid body angular velocities.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with angular velocity components
        """
        ...

    # Joint data
    def joint_rigid_body_a_sid(self) -> np.ndarray:
        """Get joint first rigid body structure IDs"""
        ...

    def joint_rigid_body_b_sid(self) -> np.ndarray:
        """Get joint second rigid body structure IDs"""
        ...

    def joint_hinge_signal_sid(self) -> np.ndarray:
        """Get joint hinge rotation angle signal structure IDs"""
        ...

    def joint_type(self) -> np.ndarray:
        """Get joint types"""
        ...

    def joint_position(self) -> np.ndarray:
        """
        Get joint positions.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with position components
        """
        ...

    def joint_axis(self) -> np.ndarray:
        """
        Get joint axes.

        Returns:
            numpy.ndarray: Array of shape [N, 3] with axis components
        """
        ...

    def joint_hinge_min(self) -> np.ndarray:
        """Get joint hinge minimum angles"""
        ...

    def joint_hinge_max(self) -> np.ndarray:
        """Get joint hinge maximum angles"""
        ...

    def joint_hinge_angle(self) -> np.ndarray:
        """Get joint hinge angles (unit: rad)"""
        ...

    def joint_hinge_torque(self) -> np.ndarray:
        """Get joint hinge torques (unit: N-m)"""
        ...

    def joint_prismatic_signal_sid(self) -> np.ndarray:
        """Get joint prismatic translation rate signal SIDs"""
        ...

    def joint_prismatic_min(self) -> np.ndarray:
        """Get joint prismatic minimum distances (unit: m)"""
        ...

    def joint_prismatic_max(self) -> np.ndarray:
        """Get joint prismatic maximum distances (unit: m)"""
        ...

    def joint_prismatic_distance(self) -> np.ndarray:
        """Get joint prismatic current distances (unit: m)"""
        ...

    def joint_prismatic_force(self) -> np.ndarray:
        """Get joint prismatic current forces (unit: N)"""
        ...


class RS_SimulationCameraRecord:
    """Interface for accessing camera simulation state"""

    def step(self) -> int:
        """Get the current simulation step"""
        ...

    def time(self) -> float:
        """Get the current simulation time"""
        ...

    def camera_gid(self) -> int:
        """Get the camera global ID"""
        ...

    def structure_gid(self) -> int:
        """Get the linked structure global ID"""
        ...

    def rigid_body_sid(self) -> int:
        """Get the linked rigid body structure ID"""
        ...

    def width(self) -> int:
        """Get the camera width in pixels"""
        ...

    def height(self) -> int:
        """Get the camera height in pixels"""
        ...

    def is_valid(self) -> bool:
        """Check if the camera record is valid"""
        ...

    def rgb(self) -> np.ndarray:
        """
        Get the camera RGB pixel data.

        Returns:
            numpy.ndarray: Array of shape [height, width, 3] with uint8 RGB color values
        """
        ...

    def position(self) -> np.ndarray:
        """Get the camera position"""
        ...

    def orientation(self) -> np.ndarray:
        """Get the camera orientation"""
        ...

    def focus_distance(self) -> np.ndarray:
        """Get the camera focus distance"""
        ...

    def field_of_view(self) -> np.ndarray:
        """Get the camera field of view"""
        ...


################################################################################
# Simulation Controllers
################################################################################

class RS_SimulationStructureController:
    """Controller for a structure in the simulation"""

    def name(self) -> str:
        """Get the structure name"""
        ...

    def structure_record(self) -> RS_SimulationStructureRecord:
        """Get the structure record (simulation state)"""
        ...

    def signal(self) -> RS_SimulationStructureSignal:
        """Get the structure signal (control interface)"""
        ...

    def camera_records(self) -> List[RS_SimulationCameraRecord]:
        """Get associated camera records"""
        ...


class RS_SimulationCameraController:
    """Controller for a camera in the simulation"""

    def name(self) -> str:
        """Get the camera name"""
        ...

    def camera_record(self) -> RS_SimulationCameraRecord:
        """Get the camera record (simulation state)"""
        ...

    def signal(self) -> RS_SimulationCameraSignal:
        """Get the camera signal (control interface)"""
        ...


class RS_SimulationForceController:
    """Controller for a force signal in the simulation"""

    def name(self) -> str:
        """Get the force signal name"""
        ...

    def signal(self) -> RS_SimulationForceSignal:
        """Get the force signal (control interface)"""
        ...


class RS_SimulationTorqueController:
    """Controller for a torque signal in the simulation"""

    def name(self) -> str:
        """Get the torque signal name"""
        ...

    def signal(self) -> RS_SimulationTorqueSignal:
        """Get the torque signal (control interface)"""
        ...


################################################################################
# Simulation Handle
################################################################################

class RS_SimulationHandle:
    """Handle for controlling a simulation instance"""

    def is_pending(self) -> bool:
        """Check if the simulation is pending"""
        ...

    def is_running(self) -> bool:
        """Check if the simulation is running"""
        ...

    def is_paused(self) -> bool:
        """Check if the simulation is paused"""
        ...

    def is_end(self) -> bool:
        """Check if the simulation is ended"""
        ...

    def pause(self) -> None:
        """Pause the simulation"""
        ...

    def resume(self) -> None:
        """Resume the simulation"""
        ...

    def terminate(self) -> None:
        """Terminate the simulation"""
        ...

    def get_status(self) -> int:
        """Get the simulation status"""
        ...

    def wait_for_running(self) -> bool:
        """Wait for the simulation to start running"""
        ...

    def wait_for_end(self) -> None:
        """Wait for the simulation to end"""
        ...

    def begin_modification_transaction(self) -> None:
        """Begin a modification transaction"""
        ...

    def end_modification_transaction(self) -> None:
        """End a modification transaction"""
        ...

    def set_simulation_config(self, config: RS_SimulationConfig) -> None:
        """Set the simulation configuration"""
        ...

    def set_world(self, config: RS_WorldConfig) -> None:
        """Set the world configuration"""
        ...

    def set_floor(self, config: RS_FloorConfig) -> None:
        """Set the floor configuration"""
        ...

    def add_material(self, config: RS_MaterialConfig) -> None:
        """Add a material to the simulation"""
        ...

    def add_record(self, config: RS_RecordConfig) -> None:
        """Add a record to the simulation"""
        ...

    def add_signal(self, config: RS_SignalConfig) -> None:
        """Add a signal to the simulation"""
        ...

    def add_structure(self, config: RS_StructureConfig) -> None:
        """Add a structure to the simulation"""
        ...

    def add_camera(self, config: RS_CameraConfig) -> None:
        """Add a camera to the simulation"""
        ...

    def update_material(self, config: RS_MaterialConfig) -> None:
        """Update a material in the simulation"""
        ...

    def delete_floor(self) -> None:
        """Delete the floor from the simulation"""
        ...

    def delete_material(self, id: int) -> None:
        """Delete a material from the simulation"""
        ...

    def delete_record(self, id: int) -> None:
        """Delete a record from the simulation"""
        ...

    def delete_signal(self, id: int) -> None:
        """Delete a signal from the simulation"""
        ...

    def delete_structure(self, id: int) -> None:
        """Delete a structure from the simulation"""
        ...

    def delete_camera(self, id: int) -> None:
        """Delete a camera from the simulation"""
        ...

    def find_material(self, name: str) -> int:
        """Find a material by name"""
        ...

    def find_record(self, name: str) -> int:
        """Find a record by name"""
        ...

    def find_signal(self, name: str) -> int:
        """Find a signal by name"""
        ...

    def find_structure(self, name: str) -> int:
        """Find a structure by name"""
        ...

    def find_camera(self, name: str) -> int:
        """Find a camera by name"""
        ...

    def get_config(self) -> RS_Config:
        """Get the full simulation configuration"""
        ...

    def get_simulation_config(self) -> RS_SimulationConfig:
        """Get the simulation configuration"""
        ...

    def get_floor_config(self) -> RS_FloorConfig:
        """Get the floor configuration"""
        ...

    def get_material_configs(self) -> List[RS_MaterialConfig]:
        """Get all material configurations"""
        ...

    def get_record_configs(self) -> List[RS_RecordConfig]:
        """Get all record configurations"""
        ...

    def get_signal_configs(self) -> List[RS_SignalConfig]:
        """Get all signal configurations"""
        ...

    def get_structure_configs(self) -> List[RS_StructureConfig]:
        """Get all structure configurations"""
        ...

    def get_camera_configs(self) -> List[RS_CameraConfig]:
        """Get all camera configurations"""
        ...


################################################################################
# Main RiseOrigin Class
################################################################################

class RiseOrigin:
    """
    Main interface for the RiseOrigin simulation system.
    """

    @overload
    def __init__(self, device: RS_DeviceLocation = ...) -> None:
        """
        Initialize a RiseOrigin simulation with the specified device.

        Args:
            device: The device location to run the simulation on. Defaults to CPU.
        """
        ...

    @overload
    def __init__(self, device_str: str = "") -> None:
        """
        Initialize a RiseOrigin simulation with the specified device string.

        Args:
            device_str: The device to use for the simulation.
                - Empty string or "cpu": Use local CPU
                - "cuda:<id>": Use CUDA device with given ID
        """
        ...

    def run_sim(
            self,
            config: Union[List[str], RS_Config],
            id: int,
            callback: Optional[
                Callable[[int, List[Tuple[RSE_SimulationControllerType, Any]]], None]
            ] = None,
    ) -> RS_SimulationHandle:
        """
        Run a simulation instance.

        Execution settings (update intervals, record buffer size, max time, save flag,
        compute/memory accuracy, log level) are read from the configuration's
        ``RS_SimulationConfig`` ``runtime_*`` fields.

        Args:
            config: Simulation configuration, either as a list of strings (file paths)
                    or as an RS_Config object.
            id: Unique simulation ID.
            callback: Optional callback function that will be called with the simulation ID and
                     controllers during simulation.

        Returns:
            Simulation handle for the running simulation.
        """
        ...
