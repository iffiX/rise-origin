// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_FILE_POOL_H
#define RS_RSC_FILE_POOL_H

#include <string>

#include "rise_origin/utils/concurrency/rs_thread_pool.h"
#include "rise_origin/utils/serialization/rs_fast_pack.h"

namespace rise {

namespace rsc {

/**
 * @brief Singleton owning the one RSC file thread pool, vending pool wired packers and unpackers.
 * @details Compression and field loading run on this shared pool. Callers take a packer or unpacker
 * by value for one save or load and let it die with the call; the pool outlives every operation.
 */
class RS_RSCFilePool {
   public:
    RS_RSCFilePool(const RS_RSCFilePool &) = delete;
    RS_RSCFilePool &operator=(const RS_RSCFilePool &) = delete;

    /**
     * @brief Get the process wide file pool, built on first use.
     */
    static RS_RSCFilePool &getInstance();

    /**
     * @brief Make an empty packer wired to the pool, its path given later at pack.
     */
    static RS_FastPacker getPacker();

    /**
     * @brief Pack a filled packer to path, stamping the RSC binary magic.
     */
    static void pack(const RS_FastPacker &packer, const std::string &path);

    /**
     * @brief Make an unpacker wired to the pool, lazily unpacked from path.
     */
    static RS_FastUnpacker getUnpacker(const std::string &path);

   private:
    /// Shared pool that compresses and loads fields in parallel.
    RS_ThreadPool _pool;

    RS_RSCFilePool() = default;

    /**
     * @brief Make an empty packer wired to the pool, its path given later at pack.
     */
    RS_FastPacker _getPacker();

    /**
     * @brief Make an unpacker wired to the pool, lazily unpacked from path.
     */
    RS_FastUnpacker _getUnpacker(const std::string &path);
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_FILE_POOL_H
