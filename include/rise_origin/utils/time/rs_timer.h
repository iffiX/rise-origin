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
     * @return The elapsed time in milliseconds.
     */
    [[nodiscard]] double elapsed() const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_time_point)
                         .time_since_epoch()
                         .count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time_point)
                       .time_since_epoch()
                       .count();

        return static_cast<double>(end - start) * 0.001;  // Convert to milliseconds
    }

    /**
     * @brief Formats the elapsed time as a string.
     *
     * @param decimal_digits The number of decimal digits to include in the string.
     * @return A formatted string representing the elapsed time.
     */
    [[nodiscard]] std::string elapsedTimeString(int decimal_digits = 3) const {
        double elapsed_time = elapsed();
        std::string unit;
        if (elapsed_time < 1.0) {
            elapsed_time *= 1000.0;
            unit = "us";
        } else if (elapsed_time < 1000.0) {
            unit = "ms";
        } else {
            elapsed_time /= 1000.0;
            unit = "s";
        }
        if (elapsed_time > 0.001) {
            return format("{:.{}f} {}", elapsed_time, decimal_digits, unit);
        }
        return "< 1 us";
    }

   private:
    /// The start time point of the timer.
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
};

}  // namespace rise

#endif  // RS_TIMER_H
