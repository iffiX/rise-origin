// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_EXEC_TRACE_H
#define RS_EXEC_TRACE_H

#include <iostream>
#include <string>
#include <string_view>

#include "rise_origin/utils/rs_env_var.h"
#include "rise_origin/utils/time/rs_timer.h"

namespace rise {

namespace device {

/// @brief True when any exec logging flag is on (build the signature only then).
inline bool isExecLogging() {
    return RS_EnvironmentVariables::isExecTraceCalls() ||
           RS_EnvironmentVariables::isExecProfilePerformance();
}

/// @brief Trim a __FILE__ path to start at the last "rise_origin" segment.
inline const char *trimExecFile(const char *file) {
    auto pos = std::string_view(file).rfind("rise_origin");
    return pos == std::string_view::npos ? file : file + pos;
}

/**
 * @brief Print one shared exec trace line "[EXEC] file(line): signature".
 * @param file Source file of the call site.
 * @param line Source line of the call site.
 * @param signature The pretty-printed call signature.
 * @param timer Optional timer; under profiling its elapsed time is appended.
 */
inline void execTracePrint(
    const char *file,
    int line,
    const std::string &signature,
    const RS_Timer *timer = nullptr
) {
    if (!isExecLogging()) return;
    if (RS_EnvironmentVariables::isExecProfilePerformance() && timer != nullptr)
        std::cout << "[EXEC] " << trimExecFile(file) << "(" << line << "): " << signature << ": "
                  << timer->elapsedTimeString() << std::endl;
    else
        std::cout << "[EXEC] " << trimExecFile(file) << "(" << line << "): " << signature
                  << std::endl;
}

/**
 * @brief Open an exec trace line before a deferred op, so a crash still shows it.
 * Under profiling the line is left open (flushed) for execTraceEnd to append the
 * elapsed time; otherwise it is terminated here.
 */
inline void execTraceBegin(const char *file, int line, const std::string &signature) {
    if (!isExecLogging()) return;
    std::cout << "[EXEC] " << trimExecFile(file) << "(" << line << "): " << signature;
    if (RS_EnvironmentVariables::isExecProfilePerformance())
        std::cout << std::flush;  // keep line open for execTraceEnd
    else
        std::cout << std::endl;
}

/// @brief Close a line opened by execTraceBegin, appending elapsed under profiling.
inline void execTraceEnd(const RS_Timer &timer) {
    if (RS_EnvironmentVariables::isExecProfilePerformance())
        std::cout << ": " << timer.elapsedTimeString() << std::endl;
}

}  // namespace device

}  // namespace rise

#endif  // RS_EXEC_TRACE_H
