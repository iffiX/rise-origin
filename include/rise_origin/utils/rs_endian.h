// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

// Adapted from https://gist.github.com/jtbr/7a43e6281e6cca353b33ee501421860c

#ifndef RS_ENDIAN_H
#define RS_ENDIAN_H

// When available, these headers can improve platform endianness detection
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
#define RS_IS_BIG_ENDIAN 1
#define RS_IS_LITTLE_ENDIAN 0
#define RS_ENDIAN_DETECTED
#elif (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || /* gcc */       \
    (defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN) /* linux header */ ||            \
    (defined(_BYTE_ORDER) && _BYTE_ORDER == _LITTLE_ENDIAN) ||                                  \
    (defined(BYTE_ORDER) && BYTE_ORDER == LITTLE_ENDIAN) /* mingw header */ ||                  \
    (defined(__sun) && defined(__SVR4) && defined(_LITTLE_ENDIAN)) || /* solaris */             \
    defined(__ARMEL__) || defined(__THUMBEL__) || defined(__AARCH64EL__) || defined(_MIPSEL) || \
    defined(__MIPSEL) || defined(__MIPSEL__) || defined(_M_IX86) || defined(_M_X64) ||          \
    defined(_M_IA64) || /* msvc for intel processors */                                         \
    defined(_M_ARM)     /* msvc code on arm executes in little endian mode */
#define RS_IS_LITTLE_ENDIAN 1
#define RS_IS_BIG_ENDIAN 0
#define RS_ENDIAN_DETECTED
#endif
#endif

#if !defined(RS_ENDIAN_DETECTED)
#error \
    "Unknown Platform / endianness. Configure endianness checks for this platform or set explicitly."
#endif

#endif  // RS_ENDIAN_H
