// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_essence.h"

void initRSPYEssence(py::module &m) {
    /***************************************************************
     * Essences
     ***************************************************************/
    rspyRSC<RSC_Material, RSC_EssenceBase>(m, "RSC_Material").def(py::init<>());

    rspyRSC<RSC_MaterialUnaryProperty, RSC_EssenceBase>(m, "RSC_MaterialUnaryProperty")
        .def_readwrite("material_reference", &RSC_MaterialUnaryProperty::material_reference);

    rspyRSC<RSC_MaterialPhysicalPropertyBase, RSC_MaterialUnaryProperty>(
        m, "RSC_MaterialPhysicalPropertyBase"
    );

    rspyRSC<RSC_MaterialPhysicalPropertyFixed, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyFixed"
    )
        .def(py::init<>());

    rspyRSC<RSC_MaterialPhysicalPropertySoft, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertySoft"
    )
        .def(py::init<>())
        .def_readwrite("elastic_modulus", &RSC_MaterialPhysicalPropertySoft::elastic_modulus)
        .def_readwrite("fail_stress", &RSC_MaterialPhysicalPropertySoft::fail_stress)
        .def_readwrite("density", &RSC_MaterialPhysicalPropertySoft::density)
        .def_readwrite("poissons_ratio", &RSC_MaterialPhysicalPropertySoft::poissons_ratio)
        .def_readwrite("min_expansion", &RSC_MaterialPhysicalPropertySoft::min_expansion)
        .def_readwrite("max_expansion", &RSC_MaterialPhysicalPropertySoft::max_expansion);

    rspyRSC<RSC_MaterialPhysicalPropertyRigid, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyRigid"
    )
        .def(py::init<>())
        .def_readwrite("density", &RSC_MaterialPhysicalPropertyRigid::density);

    rspyRSC<RSC_MaterialPhysicalPropertyAtlas, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyAtlas"
    )
        .def(py::init<>())
        .def_readwrite("mass", &RSC_MaterialPhysicalPropertyAtlas::mass)
        .def_readwrite("com_position", &RSC_MaterialPhysicalPropertyAtlas::com_position)
        .def_readwrite("inertia", &RSC_MaterialPhysicalPropertyAtlas::inertia);

    rspyRSC<RSC_MaterialOpticalPropertyBase, RSC_MaterialUnaryProperty>(
        m, "RSC_MaterialOpticalPropertyBase"
    );

    rspyRSC<RSC_MaterialOpticalPropertySimpleMicrofacet, RSC_MaterialOpticalPropertyBase>(
        m, "RSC_MaterialOpticalPropertySimpleMicrofacet"
    )
        .def(py::init<>())
        .def_readwrite(
            "albedo_texture_reference",
            &RSC_MaterialOpticalPropertySimpleMicrofacet::albedo_texture_reference
        )
        .def_readwrite(
            "roughness_metalness_texture_reference",
            &RSC_MaterialOpticalPropertySimpleMicrofacet::roughness_metalness_texture_reference
        )
        .def_readwrite(
            "is_transparent", &RSC_MaterialOpticalPropertySimpleMicrofacet::is_transparent
        )
        .def_readwrite("ior", &RSC_MaterialOpticalPropertySimpleMicrofacet::ior)
        .def_readwrite(
            "emission_texture_reference",
            &RSC_MaterialOpticalPropertySimpleMicrofacet::emission_texture_reference
        )
        .def_readwrite(
            "emission_intensity", &RSC_MaterialOpticalPropertySimpleMicrofacet::emission_intensity
        );

    rspyRSC<RSC_MaterialPairPropertyBase, RSC_EssenceBase>(m, "RSC_MaterialPairPropertyBase")
        .def_readwrite(
            "first_material_reference", &RSC_MaterialPairPropertyBase::first_material_reference
        )
        .def_readwrite(
            "second_material_reference", &RSC_MaterialPairPropertyBase::second_material_reference
        );

    rspyRSC<RSC_MaterialPairPropertyCoulombFriction, RSC_MaterialPairPropertyBase>(
        m, "RSC_MaterialPairPropertyCoulombFriction"
    )
        .def(py::init<>())
        .def_readwrite("u_static", &RSC_MaterialPairPropertyCoulombFriction::u_static)
        .def_readwrite("u_dynamic", &RSC_MaterialPairPropertyCoulombFriction::u_dynamic);

    rspyRSC<RSC_MaterialPairPropertyRestitution, RSC_MaterialPairPropertyBase>(
        m, "RSC_MaterialPairPropertyRestitution"
    )
        .def(py::init<>())
        .def_readwrite("restitution", &RSC_MaterialPairPropertyRestitution::restitution);
}
