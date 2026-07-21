// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_WORLD_SIGNAL_H
#define RSPY_INIT_WORLD_SIGNAL_H

#include "common/rspy_common.h"

/*****************************************************************
 * Signal wrappers.
 *****************************************************************/

class RSPY_WorldStructureVoxelSignal {
   public:
    RS_WorldStructureVoxelSignal signal;
    explicit RSPY_WorldStructureVoxelSignal(const RS_WorldStructureVoxelSignal &sig)
        : signal(sig) {}

    class Expansion {
       public:
        RS_WorldStructureVoxelSignal signal;
        explicit Expansion(const RS_WorldStructureVoxelSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.expansion_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> expansion() const {
            return rspyRwArray1D(signal.expansion_signals.expansion, size());
        }
    };

    class Hinge {
       public:
        RS_WorldStructureVoxelSignal signal;
        explicit Hinge(const RS_WorldStructureVoxelSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.hinge_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> hinge() const {
            return rspyRwArray1D(signal.hinge_signals.hinge, size());
        }
    };

    class Prismatic {
       public:
        RS_WorldStructureVoxelSignal signal;
        explicit Prismatic(const RS_WorldStructureVoxelSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.prismatic_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> prismatic() const {
            return rspyRwArray1D(signal.prismatic_signals.prismatic, size());
        }
    };

    [[nodiscard]] Expansion expansion_signals() const { return Expansion(signal); }
    [[nodiscard]] Hinge hinge_signals() const { return Hinge(signal); }
    [[nodiscard]] Prismatic prismatic_signals() const { return Prismatic(signal); }
};

class RSPY_WorldCameraPerspectiveSignal {
   public:
    RS_WorldCameraPerspectiveSignal signal;
    explicit RSPY_WorldCameraPerspectiveSignal(const RS_WorldCameraPerspectiveSignal &sig)
        : signal(sig) {}

    class Position {
       public:
        RS_WorldCameraPerspectiveSignal signal;
        explicit Position(const RS_WorldCameraPerspectiveSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.position_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.position_signals.position), size(), 3
            );
        }
    };

    class Orientation {
       public:
        RS_WorldCameraPerspectiveSignal signal;
        explicit Orientation(const RS_WorldCameraPerspectiveSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.orientation_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.orientation_signals.orientation), size(), 4
            );
        }
    };

    class Optics {
       public:
        RS_WorldCameraPerspectiveSignal signal;
        explicit Optics(const RS_WorldCameraPerspectiveSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.optics_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> focus_distance() const {
            return rspyRwArray1D(signal.optics_signals.focus_distance, size());
        }
        [[nodiscard]] py::array_t<RFloat> field_of_view() const {
            return rspyRwArray1D(signal.optics_signals.field_of_view, size());
        }
    };

    [[nodiscard]] Position position_signals() const { return Position(signal); }
    [[nodiscard]] Orientation orientation_signals() const { return Orientation(signal); }
    [[nodiscard]] Optics optics_signals() const { return Optics(signal); }
};

class RSPY_WorldCameraStereoSignal {
   public:
    RS_WorldCameraStereoSignal signal;
    explicit RSPY_WorldCameraStereoSignal(const RS_WorldCameraStereoSignal &sig) : signal(sig) {}

    class Position {
       public:
        RS_WorldCameraSignalBase::Position position;
        explicit Position(const RS_WorldCameraSignalBase::Position &p) : position(p) {}

        [[nodiscard]] RBigSize size() const { return position.object_num; }

        [[nodiscard]] py::array_t<RFloat> position_array() const {
            return rspyRwArray2D(reinterpret_cast<RFloat *>(position.position), size(), 3);
        }
    };

    class Orientation {
       public:
        RS_WorldCameraSignalBase::Orientation orientation;
        explicit Orientation(const RS_WorldCameraSignalBase::Orientation &o) : orientation(o) {}

        [[nodiscard]] RBigSize size() const { return orientation.object_num; }

        [[nodiscard]] py::array_t<RFloat> orientation_array() const {
            return rspyRwArray2D(reinterpret_cast<RFloat *>(orientation.orientation), size(), 4);
        }
    };

    class Optics {
       public:
        RS_WorldCameraSignalBase::Optics optics;
        explicit Optics(const RS_WorldCameraSignalBase::Optics &o) : optics(o) {}

        [[nodiscard]] RBigSize size() const { return optics.object_num; }

        [[nodiscard]] py::array_t<RFloat> field_of_view() const {
            return rspyRwArray1D(optics.field_of_view, size());
        }
    };

    [[nodiscard]] Position left_position_signals() const {
        return Position(signal.left_position_signals);
    }
    [[nodiscard]] Position right_position_signals() const {
        return Position(signal.right_position_signals);
    }
    [[nodiscard]] Orientation left_orientation_signals() const {
        return Orientation(signal.left_orientation_signals);
    }
    [[nodiscard]] Orientation right_orientation_signals() const {
        return Orientation(signal.right_orientation_signals);
    }
    [[nodiscard]] Optics left_optics_signals() const { return Optics(signal.left_optics_signals); }
    [[nodiscard]] Optics right_optics_signals() const {
        return Optics(signal.right_optics_signals);
    }
};

class RSPY_WorldRaySensorSignal {
   public:
    RS_WorldRaySensorSignal signal;
    explicit RSPY_WorldRaySensorSignal(const RS_WorldRaySensorSignal &sig) : signal(sig) {}

    class Position {
       public:
        RS_WorldRaySensorSignal signal;
        explicit Position(const RS_WorldRaySensorSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.position_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.position_signals.position), size(), 3
            );
        }
    };

    class Orientation {
       public:
        RS_WorldRaySensorSignal signal;
        explicit Orientation(const RS_WorldRaySensorSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.orientation_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.orientation_signals.orientation), size(), 4
            );
        }
    };

    [[nodiscard]] Position position_signals() const { return Position(signal); }
    [[nodiscard]] Orientation orientation_signals() const { return Orientation(signal); }
};

class RSPY_WorldForceSignal {
   public:
    RS_WorldForceSignal signal;
    explicit RSPY_WorldForceSignal(const RS_WorldForceSignal &sig) : signal(sig) {}

    class Force {
       public:
        RS_WorldForceSignal signal;
        explicit Force(const RS_WorldForceSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.force_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> force() const {
            return rspyRwArray2D(reinterpret_cast<RFloat *>(signal.force_signals.force), size(), 3);
        }
    };

    [[nodiscard]] Force force_signals() const { return Force(signal); }
};

class RSPY_WorldTorqueSignal {
   public:
    RS_WorldTorqueSignal signal;
    explicit RSPY_WorldTorqueSignal(const RS_WorldTorqueSignal &sig) : signal(sig) {}

    class Torque {
       public:
        RS_WorldTorqueSignal signal;
        explicit Torque(const RS_WorldTorqueSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.torque_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> torque() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.torque_signals.torque), size(), 3
            );
        }
    };

    [[nodiscard]] Torque torque_signals() const { return Torque(signal); }
};

class RSPY_WorldTransformSignal {
   public:
    RS_WorldTransformSignal signal;
    explicit RSPY_WorldTransformSignal(const RS_WorldTransformSignal &sig) : signal(sig) {}

    class Translation {
       public:
        RS_WorldTransformSignal signal;
        explicit Translation(const RS_WorldTransformSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.translation_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> translation() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.translation_signals.translation), size(), 3
            );
        }
    };

    class Rotation {
       public:
        RS_WorldTransformSignal signal;
        explicit Rotation(const RS_WorldTransformSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.rotation_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> rotation() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.rotation_signals.rotation), size(), 4
            );
        }

        [[nodiscard]] py::array_t<RFloat> pivot() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.rotation_signals.pivot), size(), 3
            );
        }
    };

    [[nodiscard]] Translation translation_signals() const { return Translation(signal); }

    [[nodiscard]] Rotation rotation_signals() const { return Rotation(signal); }
};

/// Bind the runtime world signal slot types.
void initRSPYWorldSignal(py::module &m);

#endif  // RSPY_INIT_WORLD_SIGNAL_H
