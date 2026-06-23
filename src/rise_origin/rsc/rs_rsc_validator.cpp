// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_validator.h"

#include <cstdint>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/string/rs_format.h"

namespace rise {

void RS_RSCValidator::validate(const rsc::RSC_Base& rsc) {
    rsc.validate(*this);
    try {
        for (auto& validator : _validators) {
            validator();
        }
    } catch (...) {
        _pool.clear();
        _pool.wait();
        _index.clear();
        _validators.clear();
        throw;
    }
    _index.clear();
    _validators.clear();
}

bool RS_RSCValidator::hasValidator(const std::string& job, const rsc::RSC_Base* dependency) const {
    return hasValidator(job, std::vector<const rsc::RSC_Base*>{dependency});
}

bool RS_RSCValidator::hasValidator(
    const std::string& job,
    const std::vector<const rsc::RSC_Base*>& dependencies
) const {
    return _index.find(_makeKey(job, dependencies)) != _index.end();
}

void RS_RSCValidator::submitValidator(
    const std::string& job,
    const rsc::RSC_Base* dependency,
    std::function<void()> validator
) {
    submitValidator(job, std::vector<const rsc::RSC_Base*>{dependency}, std::move(validator));
}

void RS_RSCValidator::submitValidator(
    const std::string& job,
    const std::vector<const rsc::RSC_Base*>& dependencies,
    std::function<void()> validator
) {
    _index.emplace(_makeKey(job, dependencies), static_cast<RIndex>(_validators.size()));
    _validators.push_back(std::move(validator));
}

void RS_RSCValidator::submitValidatorIfNotExists(
    const std::string& job,
    const rsc::RSC_Base* dependency,
    std::function<void()> validator
) {
    submitValidatorIfNotExists(
        job, std::vector<const rsc::RSC_Base*>{dependency}, std::move(validator)
    );
}

void RS_RSCValidator::submitValidatorIfNotExists(
    const std::string& job,
    const std::vector<const rsc::RSC_Base*>& dependencies,
    std::function<void()> validator
) {
    if (hasValidator(job, dependencies)) return;
    submitValidator(job, dependencies, std::move(validator));
}

std::string RS_RSCValidator::_makeKey(
    const std::string& job,
    const std::vector<const rsc::RSC_Base*>& dependencies
) {
    std::string key = job + ":[";
    for (std::size_t i = 0; i < dependencies.size(); i++) {
        if (i > 0) key += ",";
        key += format("{:x}", reinterpret_cast<std::uintptr_t>(dependencies[i]));
    }
    key += "]";
    return key;
}

}  // namespace rise
