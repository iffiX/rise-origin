// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_format.h"
/**
 * Usage example:
 * RS_DATA_EXPECT_INIT(data.provider != nullptr);
 */
#define RS_DATA_EXPECT_INIT(condition)                                             \
    if (!(condition))                                                              \
        RS_THROW(RS_UninitializedException(                                        \
            "Exception in [{}]:\nData not initialized", RS_CURRENT_FUNCTION_NAME() \
        ));

/**
 * Usage example:
 * RS_DATA_EXPECT_COPY_SRC_INIT(src_data.provider != nullptr);
 */
#define RS_DATA_EXPECT_COPY_SRC_INIT(condition)                                                \
    if (!(condition))                                                                          \
        RS_THROW(RS_UninitializedException(                                                    \
            "Exception in [{}]:\nCopy source data not initialized", RS_CURRENT_FUNCTION_NAME() \
        ));

/**
 * Usage example:
 * RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);
 */
#define RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider)                                    \
    if ((provider) == nullptr)                                                        \
        RS_THROW(RS_InvalidArgumentException(                                         \
            "Exception in [{}]:\nMemory provider is null", RS_CURRENT_FUNCTION_NAME() \
        ));

#define RS_DATA_EXPECT_POINTER_NOT_NULL(pointer)                                           \
    if ((pointer) == nullptr)                                                              \
        RS_THROW(RS_InvalidArgumentException(                                              \
            "Exception in [{}]:\nPointer {} is null", RS_CURRENT_FUNCTION_NAME(), #pointer \
        ));

/**
 * Usage example:
 * RS_DATA_EXPECT_ARGUMENT(a > 0 && b != nullptr, "A={}, B={}", a, b);
 */
#define RS_DATA_EXPECT_ARGUMENT(condition, fmt, ...)                                   \
    if (!(condition)) {                                                                \
        std::string formatted_msg = format(fmt, __VA_ARGS__);                          \
        RS_THROW(RS_InvalidArgumentException(                                          \
            "Exception in [{}]:\nInvalid argument, expect " #condition ", but get {}", \
            RS_CURRENT_FUNCTION_NAME(),                                                \
            formatted_msg                                                              \
        ));                                                                            \
    }

/**
 * Usage example:
 * RS_DATA_ASSERT_ARGUMENT(a > 0, "A must be greater than 0");
 */
#define RS_DATA_ASSERT_ARGUMENT(condition, message)                        \
    if (!(condition)) {                                                    \
        RS_THROW(RS_InvalidArgumentException(                              \
            "Exception in [{}]:\n {}", RS_CURRENT_FUNCTION_NAME(), message \
        ));                                                                \
    }
/**
 * Usage example:
 * RS_DATA_EXPECT_BOUND(index < elem_num, "index={}, elem_num={}", index, elem_num);
 */
#define RS_DATA_EXPECT_BOUND(condition, fmt, ...)                                          \
    if (!(condition)) {                                                                    \
        std::string formatted_msg = format(fmt, __VA_ARGS__);                              \
        RS_THROW(RS_OutOfBoundsException(                                                  \
            "Exception in [{}]:\nAccess out of bounds, expect " #condition ", but get {}", \
            RS_CURRENT_FUNCTION_NAME(),                                                    \
            formatted_msg                                                                  \
        ));                                                                                \
    }
