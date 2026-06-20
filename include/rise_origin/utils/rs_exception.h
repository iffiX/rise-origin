// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_EXCEPTION_H
#define RS_EXCEPTION_H

#ifndef DISABLE_THROW
#define RS_THROW(exception) throw exception
#else
#define RS_THROW(exception) void()
#endif

#include <exception>
#include <string>

#include "rise_origin/utils/string/rs_format.h"

#define RS_MAKE_EXCEPTION(exception_name, exception_base)                              \
    class exception_name : public exception_base {                                     \
       private:                                                                        \
        std::string message;                                                           \
                                                                                       \
       public:                                                                         \
        exception_name() = default;                                                    \
        explicit exception_name(const char* msg) : message(msg) {}                     \
        explicit exception_name(const std::string& msg) : message(std::move(msg)) {}   \
        template <typename... Args>                                                    \
        explicit exception_name(fmt::format_string<Args...> formatter, Args&&... args) \
            : message(fmt::format(formatter, std::forward<Args>(args)...)) {}          \
        const char* what() const noexcept override { return message.c_str(); }         \
    };

namespace rise {

// Base exception for all RiseOrigin exceptions
RS_MAKE_EXCEPTION(RS_BaseException, std::exception)

// Implementation exceptions (not implemented, deprecated, etc)
RS_MAKE_EXCEPTION(RS_ImplementationException, RS_BaseException)
RS_MAKE_EXCEPTION(RS_NotImplementedException, RS_ImplementationException)
RS_MAKE_EXCEPTION(RS_DeprecatedFeatureException, RS_ImplementationException)

// Data exceptions (out of bounds, invalid argument in function, etc)
RS_MAKE_EXCEPTION(RS_DataException, RS_BaseException)
RS_MAKE_EXCEPTION(RS_ReinitializedException, RS_DataException)
RS_MAKE_EXCEPTION(RS_UninitializedException, RS_DataException)
RS_MAKE_EXCEPTION(RS_OutOfBoundsException, RS_DataException)
RS_MAKE_EXCEPTION(RS_InvalidArgumentException, RS_DataException)
RS_MAKE_EXCEPTION(RS_InvalidCharacterizationException, RS_DataException)
RS_MAKE_EXCEPTION(RS_NotFoundException, RS_DataException)
RS_MAKE_EXCEPTION(RS_DuplicateException, RS_DataException)
RS_MAKE_EXCEPTION(RS_InvalidAccessException, RS_DataException)
RS_MAKE_EXCEPTION(RS_InvalidConversionException, RS_DataException)
RS_MAKE_EXCEPTION(RS_InvalidCastException, RS_DataException)

// Simulation exceptions (nan values, etc)
RS_MAKE_EXCEPTION(RS_SimulationException, RS_BaseException)
RS_MAKE_EXCEPTION(RS_PhysicsException, RS_SimulationException)

// Runtime exceptions (execution, memory ,communication, etc)
RS_MAKE_EXCEPTION(RS_RuntimeException, RS_BaseException)
RS_MAKE_EXCEPTION(RS_HostMemoryException, RS_RuntimeException)
RS_MAKE_EXCEPTION(RS_HostExecutionException, RS_RuntimeException)
RS_MAKE_EXCEPTION(RS_DeviceMemoryException, RS_RuntimeException)
RS_MAKE_EXCEPTION(RS_DeviceExecutionException, RS_RuntimeException)
RS_MAKE_EXCEPTION(RS_CommunicationException, RS_RuntimeException)

};  // namespace rise

#endif  // RS_EXCEPTION_H
