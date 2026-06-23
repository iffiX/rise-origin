// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_VALIDATOR_H
#define RS_RSC_VALIDATOR_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rise_origin/utils/concurrency/rs_thread_pool.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace rsc {
class RSC_Base;
}

/**
 * @brief RSC validator which holds the thread pool.
 */
class RS_RSCValidator {
   public:
    RS_RSCValidator() = default;

    /** @brief Submit the RSC's validators then run the newly registered ones in order. */
    void validate(const rsc::RSC_Base& rsc);

    /** @brief The pool an RSC submits its parallel leaf jobs to. */
    RS_ThreadPool& getThreadPool() { return _pool; }

    /** @brief True when (job, dependency) was already submitted. */
    bool hasValidator(const std::string& job, const rsc::RSC_Base* dependency) const;

    /** @brief True when (job, dependencies) was already submitted. */
    bool hasValidator(
        const std::string& job,
        const std::vector<const rsc::RSC_Base*>& dependencies
    ) const;

    /** @brief Register a validator under (job, dependency) unconditionally. */
    void submitValidator(
        const std::string& job,
        const rsc::RSC_Base* dependency,
        std::function<void()> validator
    );

    /** @brief Register a validator under (job, dependencies) unconditionally. */
    void submitValidator(
        const std::string& job,
        const std::vector<const rsc::RSC_Base*>& dependencies,
        std::function<void()> validator
    );

    /** @brief Register a validator under (job, dependency), skipping it when already submitted. */
    void submitValidatorIfNotExists(
        const std::string& job,
        const rsc::RSC_Base* dependency,
        std::function<void()> validator
    );

    /** @brief Register a validator under (job, dependencies), skipping it when already submitted.
     */
    void submitValidatorIfNotExists(
        const std::string& job,
        const std::vector<const rsc::RSC_Base*>& dependencies,
        std::function<void()> validator
    );

   private:
    /** @brief Canonical key, the job followed by the dependency pointers as job:[p1,p2,...]. */
    static std::string
    _makeKey(const std::string& job, const std::vector<const rsc::RSC_Base*>& dependencies);

    RS_ThreadPool _pool;
    std::unordered_map<std::string, RIndex> _index;
    std::vector<std::function<void()>> _validators;
};

}  // namespace rise

#endif  // RS_RSC_VALIDATOR_H
