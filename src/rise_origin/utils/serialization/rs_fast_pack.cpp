// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/utils/serialization/rs_fast_pack.h"

#include <zstd.h>

namespace rise {

namespace {

/**
 * @brief Byte shuffle that groups each byte plane of fixed width elements contiguously.
 * @note Both directions work on one block of elements, so the compressor can shuffle a field
 * a block at a time without a full field sized buffer.
 */
class RS_FastPackingShuffleFilter {
   public:
    /**
     * @brief Groups a block's element bytes into byte planes so like bytes sit together.
     * @param dst Destination planes, count times type_size bytes.
     * @param src Source element bytes of the block.
     * @param type_size The width of a single element in bytes.
     * @param count The number of elements in the block.
     */
    void forward(RByte* dst, const RByte* src, RBigSize type_size, RBigSize count) const {
        for (RBigSize k = 0; k < type_size; k++) {
            RByte* plane = dst + k * count;
            for (RBigSize i = 0; i < count; i++) plane[i] = src[i * type_size + k];
        }
    }

    /**
     * @brief Restores a block's element bytes from its byte planes.
     * @param dst Destination element bytes of the block.
     * @param src Source planes, count times type_size bytes.
     * @param type_size The width of a single element in bytes.
     * @param count The number of elements in the block.
     */
    void inverse(RByte* dst, const RByte* src, RBigSize type_size, RBigSize count) const {
        for (RBigSize k = 0; k < type_size; k++) {
            const RByte* plane = src + k * count;
            for (RBigSize i = 0; i < count; i++) dst[i * type_size + k] = plane[i];
        }
    }
};

/**
 * @brief Zstd compression for a field's bytes.
 * @note Without a filter the field is compressed in one shot into the destination buffer.
 * With a filter the field is shuffled a block at a time into a small scratch and streamed, so
 * the field is stored using only its destination buffer plus that scratch.
 */
class RS_FastPackingZstdCompression {
   public:
    /**
     * @brief Returns an upper bound on the compressed size for a raw size.
     * @param raw_size The uncompressed size in bytes.
     * @return The maximum possible compressed size in bytes.
     */
    RBigSize getMaxCompressedSize(RBigSize raw_size) const {
        return RBigSize(ZSTD_compressBound(size_t(raw_size)));
    }

    /**
     * @brief Compresses the source elements into the destination buffer.
     * @param dst Destination buffer, at least getMaxCompressedSize(raw_size) bytes.
     * @param src Source element bytes, count times type_size bytes.
     * @param type_size The width of a single element in bytes.
     * @param count The number of elements.
     * @param filter The filter applied to each block before compression, or null for none.
     * @return The number of bytes written to the destination.
     */
    RBigSize compress(
        RS_MemoryPtr dst,
        RS_ConstMemoryPtr src,
        RBigSize type_size,
        RBigSize count,
        const RS_FastPackingShuffleFilter* filter
    ) const {
        RBigSize raw_size = type_size * count;
        if (filter == nullptr) {
            size_t result =
                ZSTD_compress(dst.ptr(), dst.size(), src.ptr(), size_t(raw_size), LEVEL);
            _check(result, "compression");
            return RBigSize(result);
        }

        const RByte* in = static_cast<const RByte*>(src.ptr());
        ZSTD_CCtx* context = ZSTD_createCCtx();
        if (context == nullptr) {
            RS_THROW(RS_FastPackingException("Failed to create zstd compression context"));
        }
        ZSTD_CCtx_setParameter(context, ZSTD_c_compressionLevel, LEVEL);

        auto scratch_memory = RS_NdMemory<RByte>(
            BLOCK_ELEMENTS * type_size, &RS_MemoryDefaultProvider::getInstance(), true
        );
        RByte* scratch = scratch_memory.toMemoryPtr().as<RByte>();
        ZSTD_outBuffer output = {dst.as<RByte>(), size_t(dst.size()), 0};

        for (RBigSize base = 0; base < count; base += BLOCK_ELEMENTS) {
            RBigSize block_count = RS_MIN(BLOCK_ELEMENTS, count - base);
            filter->forward(scratch, in + base * type_size, type_size, block_count);
            ZSTD_inBuffer input = {scratch, size_t(block_count * type_size), 0};
            while (input.pos < input.size) {
                size_t result = ZSTD_compressStream2(context, &output, &input, ZSTD_e_continue);
                if (ZSTD_isError(result)) {
                    ZSTD_freeCCtx(context);
                    RS_THROW(RS_FastPackingException(
                        "Zstd stream compression failed: {}", ZSTD_getErrorName(result)
                    ));
                }
            }
        }

        for (;;) {
            ZSTD_inBuffer input = {nullptr, 0, 0};
            size_t remaining = ZSTD_compressStream2(context, &output, &input, ZSTD_e_end);
            if (ZSTD_isError(remaining)) {
                ZSTD_freeCCtx(context);
                RS_THROW(RS_FastPackingException(
                    "Zstd stream flush failed: {}", ZSTD_getErrorName(remaining)
                ));
            }
            if (remaining == 0) break;
        }

        ZSTD_freeCCtx(context);
        return RBigSize(output.pos);
    }

    /**
     * @brief Decompresses the source bytes into the destination buffer.
     * @param dst Destination buffer, exactly raw_size bytes.
     * @param src Stored source bytes.
     * @param stored_size The size of the stored source bytes.
     * @param raw_size The expected uncompressed size in bytes.
     * @param type_size The width of a single element in bytes.
     * @param count The number of elements.
     * @param filter The filter that was applied to each block, or null for none.
     */
    void decompress(
        RS_MemoryPtr dst,
        RS_ConstMemoryPtr src,
        RBigSize stored_size,
        RBigSize raw_size,
        RBigSize type_size,
        RBigSize count,
        const RS_FastPackingShuffleFilter* filter
    ) const {
        if (filter == nullptr) {
            size_t result =
                ZSTD_decompress(dst.ptr(), size_t(raw_size), src.ptr(), size_t(stored_size));
            _check(result, "decompression");
            if (RBigSize(result) != raw_size) {
                RS_THROW(RS_FastPackingException(
                    "Zstd decompression size mismatch: expected {}, got {}",
                    raw_size,
                    RBigSize(result)
                ));
            }
            return;
        }

        const RByte* in = static_cast<const RByte*>(src.ptr());
        RByte* out = dst.as<RByte>();
        ZSTD_DCtx* context = ZSTD_createDCtx();
        if (context == nullptr) {
            RS_THROW(RS_FastPackingException("Failed to create zstd decompression context"));
        }

        auto scratch_memory = RS_NdMemory<RByte>(
            BLOCK_ELEMENTS * type_size, &RS_MemoryDefaultProvider::getInstance(), true
        );
        RByte* scratch = scratch_memory.toMemoryPtr().as<RByte>();
        ZSTD_inBuffer input = {in, size_t(stored_size), 0};

        for (RBigSize base = 0; base < count; base += BLOCK_ELEMENTS) {
            RBigSize block_count = RS_MIN(BLOCK_ELEMENTS, count - base);
            ZSTD_outBuffer output = {scratch, size_t(block_count * type_size), 0};
            while (output.pos < output.size) {
                size_t result = ZSTD_decompressStream(context, &output, &input);
                if (ZSTD_isError(result)) {
                    ZSTD_freeDCtx(context);
                    RS_THROW(RS_FastPackingException(
                        "Zstd stream decompression failed: {}", ZSTD_getErrorName(result)
                    ));
                }
                if (output.pos < output.size && input.pos == input.size) {
                    ZSTD_freeDCtx(context);
                    RS_THROW(
                        RS_FastPackingException("Zstd stream ended before the field was complete")
                    );
                }
            }
            filter->inverse(out + base * type_size, scratch, type_size, block_count);
        }

        ZSTD_freeDCtx(context);
    }

   private:
    /// Zstd compression level used for every field.
    static constexpr int LEVEL = 3;

    /// Number of elements per shuffle block, chosen so the scratch buffer stays small.
    static constexpr RBigSize BLOCK_ELEMENTS = RBigSize(1) << 16;

    /**
     * @brief Throws when a zstd result code reports an error.
     * @param result The zstd result code.
     * @param action The action name used in the message.
     */
    static void _check(size_t result, const char* action) {
        if (ZSTD_isError(result)) {
            RS_THROW(
                RS_FastPackingException("Zstd {} failed: {}", action, ZSTD_getErrorName(result))
            );
        }
    }
};

}  // namespace

RS_FastPackingCompressor::RS_FastPackingCompressor(
    RSE_FastPackingCompressionMethod method,
    RSE_FastPackingCompressionFilter filter
)
    : _method(method), _filter(filter) {
    if (_filter != RSE_FAST_PACKING_COMPRESSION_FILTER_NONE &&
        _method == RSE_FAST_PACKING_COMPRESSION_METHOD_NONE) {
        RS_THROW(RS_FastPackingException(
            "A fast packing filter requires a compression method, filter [{}] was set with method "
            "none",
            int(_filter)
        ));
    }
}

RBigSize RS_FastPackingCompressor::getMaxCompressedSize(RBigSize raw_size) const {
    if (_method == RSE_FAST_PACKING_COMPRESSION_METHOD_ZSTD) {
        return RS_FastPackingZstdCompression().getMaxCompressedSize(raw_size);
    }
    return raw_size;
}

RBigSize RS_FastPackingCompressor::compress(
    RS_MemoryPtr dst,
    RS_ConstMemoryPtr src,
    RBigSize type_size,
    RBigSize count
) const {
    RBigSize raw_size = type_size * count;
    if (_method == RSE_FAST_PACKING_COMPRESSION_METHOD_NONE) {
        dst.copyFrom(src, raw_size);
        return raw_size;
    }
    RS_FastPackingShuffleFilter shuffle;
    const RS_FastPackingShuffleFilter* filter =
        (_filter == RSE_FAST_PACKING_COMPRESSION_FILTER_SHUFFLE && type_size > 1) ? &shuffle
                                                                                  : nullptr;
    return RS_FastPackingZstdCompression().compress(dst, src, type_size, count, filter);
}

void RS_FastPackingCompressor::decompress(
    RS_MemoryPtr dst,
    RS_ConstMemoryPtr src,
    RBigSize stored_size,
    RBigSize raw_size,
    RBigSize type_size,
    RBigSize count
) const {
    if (_method == RSE_FAST_PACKING_COMPRESSION_METHOD_NONE) {
        dst.copyFrom(src, raw_size);
        return;
    }
    RS_FastPackingShuffleFilter shuffle;
    const RS_FastPackingShuffleFilter* filter =
        (_filter == RSE_FAST_PACKING_COMPRESSION_FILTER_SHUFFLE && type_size > 1) ? &shuffle
                                                                                  : nullptr;
    RS_FastPackingZstdCompression().decompress(
        dst, src, stored_size, raw_size, type_size, count, filter
    );
}

}  // namespace rise
