// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SYSTEM_H
#define RS_SYSTEM_H

#include <array>
#include <cstdint>
#include <string>

#include "rise_origin/utils/rs_def.h"

// System type detection
#if defined(__linux) || defined(__linux__)
#define RS_SYSTEM_IS_LINUX
#elif defined(__APPLE__)
#define RS_SYSTEM_IS_DARWIN
#elif defined(_WIN32)
#define RS_SYSTEM_IS_WINDOWS
#endif

#if defined(RS_SYSTEM_IS_LINUX) || defined(RS_SYSTEM_IS_DARWIN)
#include <unistd.h>
#elif defined(RS_SYSTEM_IS_WINDOWS)
#include <windows.h>
#endif

// Compiler detection
#if defined(__GNUC__)
#define RS_COMPILER_IS_GCC
#elif defined(__clang__)
#define RS_COMPILER_IS_CLANG
#elif defined(_MSC_VER)
#define RS_COMPILER_IS_MSVC
#endif

// System endianness detection
// Adapted from https://gist.github.com/jtbr/7a43e6281e6cca353b33ee501421860c
#ifdef __has_include  // C++17, supported as extension to C++11 in clang, GCC 5+, vs2015
#if __has_include(<endian.h>)
#include <endian.h>  // gnu libc normally provides, linux
#elif __has_include(<machine/endian.h>)
#include <machine/endian.h>  //open bsd, macos
#elif __has_include(<sys/param.h>)
#include <sys/param.h>  // mingw, some bsd (not open/macos)
#elif __has_include(<sys/isadefs.h>)
#include <sys/isadefs.h>  // solaris
#endif
#endif

#if !defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)
#if (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) ||                     \
    (defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN) ||                                 \
    (defined(_BYTE_ORDER) && _BYTE_ORDER == _BIG_ENDIAN) ||                                    \
    (defined(BYTE_ORDER) && BYTE_ORDER == BIG_ENDIAN) ||                                       \
    (defined(__sun) && defined(__SVR4) && defined(_BIG_ENDIAN)) || defined(__ARMEB__) ||       \
    defined(__THUMBEB__) || defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) || \
    defined(__MIBSEB__) || defined(_M_PPC)
#define RS_SYSTEM_IS_BIG_ENDIAN 1
#define RS_SYSTEM_IS_LITTLE_ENDIAN 0
#define RS_SYSTEM_IS_ENDIAN_DETECTED
#elif (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || /* gcc */       \
    (defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN) /* linux header */ ||            \
    (defined(_BYTE_ORDER) && _BYTE_ORDER == _LITTLE_ENDIAN) ||                                  \
    (defined(BYTE_ORDER) && BYTE_ORDER == LITTLE_ENDIAN) /* mingw header */ ||                  \
    (defined(__sun) && defined(__SVR4) && defined(_LITTLE_ENDIAN)) || /* solaris */             \
    defined(__ARMEL__) || defined(__THUMBEL__) || defined(__AARCH64EL__) || defined(_MIPSEL) || \
    defined(__MIPSEL) || defined(__MIPSEL__) || defined(_M_IX86) || defined(_M_X64) ||          \
    defined(_M_IA64) || /* msvc for intel processors */                                         \
    defined(_M_ARM)     /* msvc code on arm executes in little endian mode */
#define RS_SYSTEM_IS_LITTLE_ENDIAN 1
#define RS_SYSTEM_IS_BIG_ENDIAN 0
#define RS_SYSTEM_IS_ENDIAN_DETECTED
#endif
#endif

namespace rise {

RS_ENUM(
    RSE_SystemType,
    RIndex,
    (RSE_SYSTEM_TYPE_UNKNOWN, 0),
    (RSE_SYSTEM_TYPE_LINUX, 1),
    (RSE_SYSTEM_TYPE_DARWIN, 2),
    (RSE_SYSTEM_TYPE_WINDOWS, 3)
);

RS_ENUM(
    RSE_SystemEndianness,
    RIndex,
    (RSE_SYSTEM_ENDIANNESS_UNKNOWN, 0),
    (RSE_SYSTEM_ENDIANNESS_LITTLE, 1),
    (RSE_SYSTEM_ENDIANNESS_BIG, 2)
);

class RS_System {
   public:
    static constexpr RSE_SystemType getSystemType() {
#ifdef RS_SYSTEM_IS_LINUX
        return RSE_SYSTEM_TYPE_LINUX;
#elif RS_SYSTEM_IS_DARWIN
        return RSE_SYSTEM_TYPE_DARWIN;
#elif RS_SYSTEM_IS_WINDOWS
        return RSE_SYSTEM_TYPE_WINDOWS;
#else
        return RSE_SYSTEM_TYPE_UNKNOWN;
#endif
    }

    static constexpr RSE_SystemEndianness getSystemEndianness() {
#if RS_SYSTEM_IS_LITTLE_ENDIAN
        return RSE_SYSTEM_ENDIANNESS_LITTLE;
#elif RS_SYSTEM_IS_BIG_ENDIAN
        return RSE_SYSTEM_ENDIANNESS_BIG;
#else
        return RSE_SYSTEM_ENDIANNESS_UNKNOWN;
#endif
    }

    static std::string getHostName() {
#if defined(RS_SYSTEM_IS_LINUX) || defined(RS_SYSTEM_IS_DARWIN)
        std::array<char, 256> host_name{};
        if (::gethostname(host_name.data(), host_name.size()) != 0) {
            return "";
        }
        host_name.back() = '\0';
        return std::string(host_name.data());
#elif defined(RS_SYSTEM_IS_WINDOWS)
        std::array<char, MAX_COMPUTERNAME_LENGTH + 1> host_name{};
        DWORD host_name_size = static_cast<DWORD>(host_name.size());
        if (!::GetComputerNameA(host_name.data(), &host_name_size)) {
            return "";
        }
        return std::string(host_name.data(), static_cast<std::size_t>(host_name_size));
#else
        return "";
#endif
    }

    static RBigIndex getProcessId() {
#if defined(RS_SYSTEM_IS_LINUX) || defined(RS_SYSTEM_IS_DARWIN)
        return static_cast<RBigIndex>(::getpid());
#elif defined(RS_SYSTEM_IS_WINDOWS)
        return static_cast<RBigIndex>(::GetCurrentProcessId());
#else
        return 0;
#endif
    }
};

}  // namespace rise

#endif  // RS_SYSTEM_H
