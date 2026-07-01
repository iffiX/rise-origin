// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_TIMER_H
#define RS_TIMER_H

#include <chrono>

#include "rise_origin/utils/string/rs_format.h"

namespace rise {

/**
 * @brief Class representing a high-resolution timer.
 *
 * This timer allows measuring elapsed time with high precision.
 */
class RS_Timer {
   public:
    /**
     * @brief Constructs a timer and sets the start time point.
     */
    RS_Timer() : start_time_point(std::chrono::high_resolution_clock::now()) {}

    /**
     * @brief Resets the timer by updating the start time point.
     */
    void reset() { start_time_point = std::chrono::high_resolution_clock::now(); }

    /**
     * @brief Calculates the elapsed time since the timer was started.
     *
     * @return The elapsed time in seconds.
     */
    [[nodiscard]] double elapsed() const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_time_point)
                         .time_since_epoch()
                         .count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time_point)
                       .time_since_epoch()
                       .count();

        return static_cast<double>(end - start) * 1e-6;  // Convert to seconds
    }

    /**
     * @brief Formats the elapsed time as a string.
     *
     * @param decimal_digits The number of decimal digits to include in the string.
     * @return A formatted string representing the elapsed time.
     */
    [[nodiscard]] std::string elapsedTimeString(int decimal_digits = 3) const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time_point - start_time_point)
                .count();
        if (elapsed_us < 1) {
            return "< 1 us";
        }
        if (elapsed_us < 1000) {
            return format("{:.{}f} us", static_cast<double>(elapsed_us), decimal_digits);
        }
        if (elapsed_us < 1000000) {
            return format("{:.{}f} ms", static_cast<double>(elapsed_us) / 1000.0, decimal_digits);
        }
        return format("{:.{}f} s", static_cast<double>(elapsed_us) / 1000000.0, decimal_digits);
    }

   private:
    /// The start time point of the timer.
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
};

}  // namespace rise

#endif  // RS_TIMER_H
