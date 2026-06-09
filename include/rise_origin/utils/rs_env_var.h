// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ENV_H
#define RS_ENV_H

#include <cstdlib>
#include <iostream>
#include <string>

namespace rise {

// Singleton class
class RS_EnvironmentVariables {
   public:
    static RS_EnvironmentVariables& getInstance() {
        static RS_EnvironmentVariables instance;
        return instance;
    }

    RS_EnvironmentVariables(const RS_EnvironmentVariables&) = delete;

    RS_EnvironmentVariables& operator=(const RS_EnvironmentVariables&) = delete;

    bool isExecForceSync() const { return _is_exec_force_sync; }

    bool isExecTraceCalls() const { return _is_exec_trace_calls; }

    bool isExecProfilePerformance() const { return _is_exec_profile_performance; }

   private:
    RS_EnvironmentVariables() {
        const char* exec_force_sync = std::getenv("EXEC_FORCE_SYNC");
        if (exec_force_sync != nullptr && std::string(exec_force_sync) == "1") {
            std::cout << "EXEC_FORCE_SYNC set to true." << std::endl;
            _is_exec_force_sync = true;
        }

        const char* exec_trace_calls = std::getenv("EXEC_TRACE_CALLS");
        if (exec_trace_calls != nullptr && std::string(exec_trace_calls) == "1") {
            std::cout << "EXEC_TRACE_CALLS set to true." << std::endl;
            _is_exec_trace_calls = true;
        }

        const char* exec_profile_performance = std::getenv("EXEC_PROFILE_PERFORMANCE");
        if (exec_profile_performance != nullptr && std::string(exec_profile_performance) == "1") {
            std::cout << "EXEC_PROFILE_PERFORMANCE set to true." << std::endl;
            _is_exec_profile_performance = true;
        }

        // Trace and profile imply sync, so the printed signatures and timings are accurate.
        if (_is_exec_trace_calls || _is_exec_profile_performance) _is_exec_force_sync = true;
    }

    bool _is_exec_force_sync = false;
    bool _is_exec_trace_calls = false;
    bool _is_exec_profile_performance = false;
};

}  // namespace rise

#endif  // RS_ENV_H
