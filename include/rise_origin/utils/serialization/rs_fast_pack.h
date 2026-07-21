// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_FAST_PACK_H
#define RS_FAST_PACK_H

#include <fmt/ranges.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rise_origin/utils/concurrency/rs_thread_pool.h"
#include "rise_origin/utils/data/rs_any.h"
#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/memory/rs_nd_memory.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/rs_system.h"
#include "rise_origin/utils/serialization/rs_serialization_helper.h"
#include "rise_origin/utils/string/rs_format.h"

#define RS_PACKING_META_FIELD_NAME "meta"
#define RS_PACKING_NAME_MAX_LEN 32
#define RS_PACKING_SHAPE_MAX_LEN 4

namespace rise {

/*****************************************************************
 * Exceptions
 *****************************************************************/
RS_MAKE_EXCEPTION(RS_FastPackingException, RS_DataException);

/*****************************************************************
 * Enums
 *****************************************************************/
/**
 * @brief Compression applied to a field's stored bytes.
 */
RS_ENUM(
    RSE_FastPackingCompressionMethod,
    uint8_t,
    (RSE_FAST_PACKING_COMPRESSION_METHOD_NONE, 0),
    (RSE_FAST_PACKING_COMPRESSION_METHOD_ZSTD, 1)
);

/**
 * @brief Filter applied to a field's elements before compression.
 */
RS_ENUM(
    RSE_FastPackingCompressionFilter,
    uint8_t,
    (RSE_FAST_PACKING_COMPRESSION_FILTER_NONE, 0),
    (RSE_FAST_PACKING_COMPRESSION_FILTER_SHUFFLE, 1)
);

/*****************************************************************
 * Helper Functions
 *****************************************************************/
/**
 * @brief Checks if the platform has the float and integer layout required for fast packing.
 * @return True if the platform has the required IEEE 754 float layout and binary integers.
 */
constexpr bool isFastPackingValidFloatLayout() {
    return std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::radix == 2 &&
           std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::radix == 2 &&
           std::numeric_limits<int>::radix == 2;
}

/**
 * @brief Checks if the host requires byte swapping to read or write the little endian wire format.
 * @return True if the host is not little endian.
 */
inline bool isFastPackingByteSwapNeeded() {
    return RS_System::getSystemEndianness() != RSE_SYSTEM_ENDIANNESS_LITTLE;
}

/**
 * @brief Byte-swaps each element of a buffer in place, keyed on the element width.
 * @param data The buffer to swap.
 * @param count The number of elements.
 * @param data_size The width of each element in bytes.
 */
inline void byteSwapFastPackingElements(RS_MemoryPtr data, RBigSize count, uint32_t data_size) {
    using namespace serialization_helper;
    if (data_size == 2) {
        auto* p = data.as<uint16_t>();
        for (RBigSize i = 0; i < count; i++) p[i] = bswap16(p[i]);
    } else if (data_size == 4) {
        auto* p = data.as<uint32_t>();
        for (RBigSize i = 0; i < count; i++) p[i] = bswap32(p[i]);
    } else if (data_size == 8) {
        auto* p = data.as<uint64_t>();
        for (RBigSize i = 0; i < count; i++) p[i] = bswap64(p[i]);
    }
}

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief Structure representing a field type in packing.
 */
struct RS_FastPackingFieldType {
    /// Data type of the field ('i' for integer, 'u' for unsigned, 'f' for float, 's' for struct).
    char data_type;
    /// Reserved
    uint8_t _reserved[3] = {0, 0, 0};
    /// Size of the data type in bytes.
    uint32_t data_size;

    constexpr RS_FastPackingFieldType() : data_type(' '), data_size(0) {};

    constexpr RS_FastPackingFieldType(char data_type, uint32_t data_size)
        : data_type(data_type), data_size(data_size) {};

    /**
     * @brief Determines the field type for a data type.
     * @tparam T The data type.
     * @return The field type.
     */
    template <typename T>
    static constexpr RS_FastPackingFieldType getType() {
        RS_FastPackingFieldType type;
        if constexpr (std::is_same_v<T, int64_t>) {
            type.data_type = 'i';
            type.data_size = 8;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            type.data_type = 'u';
            type.data_size = 8;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            type.data_type = 'i';
            type.data_size = 4;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            type.data_type = 'u';
            type.data_size = 4;
        } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, short>) {
            type.data_type = 'i';
            type.data_size = 2;
        } else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, unsigned short>) {
            type.data_type = 'u';
            type.data_size = 2;
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, char>) {
            type.data_type = 'i';
            type.data_size = 1;
        } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, unsigned char>) {
            type.data_type = 'u';
            type.data_size = 1;
        } else if constexpr (std::is_same_v<T, float>) {
            type.data_type = 'f';
            type.data_size = 4;
        } else if constexpr (std::is_same_v<T, double>) {
            type.data_type = 'f';
            type.data_size = 8;
        } else if constexpr (std::is_class_v<T>) {
            type.data_type = 's';
            type.data_size = sizeof(T);
        } else {
            type.data_type = '0';
            type.data_size = 0;
        }
        return type;
    }

    bool operator==(const RS_FastPackingFieldType& type) const {
        return data_type == type.data_type && data_size == type.data_size;
    }
    bool operator!=(const RS_FastPackingFieldType& type) const { return !(*this == type); }
};

#pragma pack(push, 1)
/**
 * @brief Structure representing a row in the packing header.
 */
struct RS_FastPackingHeaderRow {
    /// Name of the field.
    char name[RS_PACKING_NAME_MAX_LEN];
    /// Type of a single element.
    RS_FastPackingFieldType type;

    /// Number of valid dimensions in shape.
    uint64_t dimensions;
    /// Shape of the field, row-major, first dimensions entries are valid.
    uint64_t shape[RS_PACKING_SHAPE_MAX_LEN];
    /// Total number of elements, the product of the valid shape dimensions.
    uint64_t size;

    /// Size of the uncompressed field data in bytes, size times type.data_size.
    uint64_t raw_size;
    /// Size of the stored field bytes after filter and compression, equal to raw_size when both are
    /// none.
    uint64_t stored_size;
    /// Byte offset of the stored field bytes from the packing header start.
    uint64_t offset;

    /// Compression applied to the stored bytes.
    RSE_FastPackingCompressionMethod compression;
    /// Filter applied to the elements before compression.
    RSE_FastPackingCompressionFilter filter;
    /// Reserved for future codec parameters.
    uint8_t _reserved[6];

    /**
     * @brief Byte-swaps the integer members in place, converting between host and wire order.
     */
    void byteSwap() {
        using namespace serialization_helper;
        type.data_size = bswap32(type.data_size);
        dimensions = bswap64(dimensions);
        for (int dim = 0; dim < RS_PACKING_SHAPE_MAX_LEN; dim++) shape[dim] = bswap64(shape[dim]);
        size = bswap64(size);
        raw_size = bswap64(raw_size);
        stored_size = bswap64(stored_size);
        offset = bswap64(offset);
    }
};

/**
 * @brief Structure representing the packing header.
 */
struct RS_FastPackingHeader {
    /// Number of rows in the header.
    uint32_t row_num;
    /// Array of header rows.
    RS_FastPackingHeaderRow rows[];
};
#pragma pack(pop)

/*****************************************************************
 * Compression
 *****************************************************************/
/**
 * @brief Codec that filters then compresses a field's bytes under a fixed method and filter.
 * @note The filter and the method are fused, so a field is stored using only its compressed
 * output buffer plus a small internal scratch. A filter requires a compression method, so
 * constructing a filter with the none method throws. The method and filter are defined in
 * rs_fast_pack.cpp, which owns the zstd and shuffle machinery.
 */
class RS_FastPackingCompressor {
   public:
    /**
     * @brief Constructs a compressor for a method and filter.
     * @param method The compression method applied to a field's bytes.
     * @param filter The filter applied to a field's elements before compression.
     * @throws RS_FastPackingException if filter is set while method is none.
     */
    RS_FastPackingCompressor(
        RSE_FastPackingCompressionMethod method,
        RSE_FastPackingCompressionFilter filter
    );

    /**
     * @brief Returns an upper bound on the stored size for a raw size.
     * @param raw_size The uncompressed size in bytes.
     * @return The maximum possible stored size in bytes.
     */
    [[nodiscard]] RBigSize getMaxCompressedSize(RBigSize raw_size) const;

    /**
     * @brief Filters then compresses the source elements into the destination buffer.
     * @param dst Destination buffer, at least getMaxCompressedSize(raw_size) bytes.
     * @param src Source element bytes, count times type_size bytes.
     * @param type_size The width of a single element in bytes.
     * @param count The number of elements.
     * @return The number of bytes written to the destination.
     */
    RBigSize
    compress(RS_MemoryPtr dst, RS_ConstMemoryPtr src, RBigSize type_size, RBigSize count) const;

    /**
     * @brief Decompresses then inverse filters the source bytes into the destination buffer.
     * @param dst Destination buffer, exactly raw_size bytes.
     * @param src Stored source bytes.
     * @param stored_size The size of the stored source bytes.
     * @param raw_size The expected uncompressed size in bytes, count times type_size.
     * @param type_size The width of a single element in bytes.
     * @param count The number of elements.
     */
    void decompress(
        RS_MemoryPtr dst,
        RS_ConstMemoryPtr src,
        RBigSize stored_size,
        RBigSize raw_size,
        RBigSize type_size,
        RBigSize count
    ) const;

   private:
    /// The compression method applied to a field's bytes.
    RSE_FastPackingCompressionMethod _method;

    /// The filter applied to a field's elements before compression.
    RSE_FastPackingCompressionFilter _filter;
};

/**
 * @brief Fast packer, designed for packing gigabytes of data in less than a second,
 * minimizes memory copy overhead. It is able to pack from multiple memory fields from
 * multiple devices to the target memory region with minimal memory copy overhead.
 *
 */
class RS_FastPacker {
   public:
    /**
     * @brief Structure representing an iterator for packing data.
     */
    struct RS_FastPackerIterator {
        /// Pointer to the packer instance.
        const RS_FastPacker* packer;

        /// Header data for packing.
        RS_NdMemory<RByte> header;

        /// Current header pointer up to the place it has been packed.
        RS_MemoryPtr header_ptr = {};

        /// Current header remaining copy size.
        RBigSize header_remaining_copy_size = 0;

        /// Current field index.
        RIndex field = RS_NULL_INDEX;

        /// Current field remaining copy size.
        RBigSize field_remaining_copy_size = 0;

        /// Current field pointer up to the place it has been packed.
        RS_MemoryPtr field_ptr = {};

        RS_FastPackerIterator() = delete;

        /**
         * @brief Constructs a packer iterator.
         *
         * @param packer Pointer to the packer instance.
         */
        explicit RS_FastPackerIterator(const RS_FastPacker* packer) : packer(packer) {
            auto& fields = packer->_fields;
            header = RS_NdMemory<RByte>(packer->_getHeaderSize(), packer->_provider, true);
            packer->_writeHeader(header.toMemoryPtr());

            header_ptr = header.toMemoryPtr();
            // Header ptr always exactly matches the header size
            header_remaining_copy_size = header_ptr.size();

            if (!fields.empty()) {
                field = 0;
                field_ptr = packer->_stored_ptrs[0];
                field_remaining_copy_size = fields[0].second.stored_size;
            }
        }

        /**
         * @return True if the iterator has remaining data to read, false if end is reached.
         */
        explicit operator bool() const {
            // Either the current field has remaining data
            // or the current field is an empty field within valid field index bound
            return !header_ptr.empty() ||
                   (field != RS_NULL_INDEX && field < packer->_fields.size());
        }

        /**
         * @brief Reads data from the packer into the packing target.
         *
         * @param dst Destination buffer to store the read data.
         * @return The size of data read.
         * @throws RS_InvalidArgumentException if the destination buffer is empty.
         */
        RBigSize read(RS_MemoryPtr dst) {
            if (dst.empty()) {
                RS_THROW(RS_InvalidArgumentException("Destination buffer is empty"));
            }
            RBigSize copied_size = 0;

            if (header_remaining_copy_size > 0) {
                RBigSize new_copied_size = _packCopy(dst, header_ptr, header_remaining_copy_size);
                copied_size += new_copied_size;
                header_remaining_copy_size -= new_copied_size;
                if (dst.empty()) return copied_size;
            }

            while (!dst.empty() && field < packer->_fields.size()) {
                // Copying current field, then move to the next field
                RBigSize new_copied_size = _packCopy(dst, field_ptr, field_remaining_copy_size);
                copied_size += new_copied_size;
                field_remaining_copy_size -= new_copied_size;
                if (dst.empty()) return copied_size;

                if (field_remaining_copy_size == 0) {
                    field++;
                    if (field < packer->_fields.size()) {
                        field_ptr = packer->_stored_ptrs[field];
                        field_remaining_copy_size = packer->_fields[field].second.stored_size;
                    } else {
                        field_ptr = {};
                        field_remaining_copy_size = 0;
                    }
                }
            }
            return copied_size;
        }

       private:
        /**
         * @brief Copies data from source to destination. Updates the destination and
         * source pointers.
         *
         * @param dst Destination buffer.
         * @param src Source buffer.
         * @param copy_size Size of the data to copy.
         * @return The size of the data copied.
         */
        static RBigSize _packCopy(RS_MemoryPtr& dst, RS_MemoryPtr& src, RBigSize copy_size) {
            RBigSize pack_size = RS_MIN(dst.size(), copy_size);
            dst.copyFrom(src, pack_size);
            dst += pack_size;
            src += pack_size;
            return pack_size;
        }
    };

    /**
     * @brief Constructs a fast packer.
     * @param mem_provider The memory provider to use.
     * @param pool The thread pool used to compress fields in parallel, or null for serial.
     */
    explicit RS_FastPacker(
        RS_MemoryProvider* mem_provider = &RS_MemoryDefaultProvider::getInstance(),
        RS_ThreadPool* pool = nullptr
    ) {
        static_assert(
            isFastPackingValidFloatLayout(),
            "The platform must have the required IEEE 754 float layout"
        );
        _provider = mem_provider;
        _pool = pool;
        _is_byte_swap_needed = isFastPackingByteSwapNeeded();
    }

    RS_FastPacker(const RS_FastPacker& other)
        : _provider(other._provider),
          _pool(other._pool),
          _is_byte_swap_needed(other._is_byte_swap_needed),
          _fields(other._fields),
          _buffers(other._buffers) {}

    RS_FastPacker& operator=(const RS_FastPacker& other) {
        _provider = other._provider;
        _pool = other._pool;
        _is_byte_swap_needed = other._is_byte_swap_needed;
        _fields = other._fields;
        _buffers = other._buffers;
        _invalidateStored();
        return *this;
    }

    RS_FastPacker(RS_FastPacker&& other) noexcept {
        _provider = other._provider;
        _pool = other._pool;
        _is_byte_swap_needed = other._is_byte_swap_needed;
        _fields = std::move(other._fields);
        _buffers = std::move(other._buffers);
        _invalidateStored();
        other._invalidateStored();
    }

    RS_FastPacker& operator=(RS_FastPacker&& other) noexcept {
        _provider = other._provider;
        _pool = other._pool;
        _is_byte_swap_needed = other._is_byte_swap_needed;
        _fields = std::move(other._fields);
        _buffers = std::move(other._buffers);
        _invalidateStored();
        other._invalidateStored();
        return *this;
    }

    /**
     * @brief Sets the meta-data of the packer.
     * @note The meta-data is stored in the first field of the packer. Setting meta-data
     * will override the existing meta-data, and invalidate any iterator of the packer.
     * @param meta_data The meta-data to set.
     * @param compression The compression applied to the meta-data bytes.
     */
    void setMetaData(
        const RS_JSON& meta_data,
        RSE_FastPackingCompressionMethod compression = RSE_FAST_PACKING_COMPRESSION_METHOD_NONE
    ) {
        std::string meta_data_str = meta_data.dump();

        // Note: include '\0' in the serialized data
        RBigSize real_size = meta_data_str.size() + 1;
        auto p = RS_NdMemory<RByte>(real_size, _provider, true);
        p.writeSlice(reinterpret_cast<const RByte*>(meta_data_str.data()), meta_data_str.size(), 0);
        p[meta_data_str.size()] = uint8_t('\0');

        RS_FastPackingHeaderRow row;
        memset(&row, 0, sizeof(row));
        strncpy(row.name, RS_PACKING_META_FIELD_NAME, RS_PACKING_NAME_MAX_LEN - 1);
        row.type = RS_FastPackingFieldType::getType<char>();
        row.dimensions = 1;
        row.shape[0] = real_size;
        row.size = real_size;
        row.raw_size = real_size;
        row.compression = compression;
        row.filter = RSE_FAST_PACKING_COMPRESSION_FILTER_NONE;

        // Add meta-data field to the front of the fields
        if (_fields.empty() || std::string(_fields[0].second.name) != RS_PACKING_META_FIELD_NAME) {
            _buffers.insert(_buffers.begin(), RS_Any(p));
            _fields.insert(_fields.begin(), {p.toMemoryPtr(), row});
        } else {
            _buffers[0] = RS_Any(p);
            _fields[0] = {p.toMemoryPtr(), row};
        }
        _invalidateStored();
    }

    /**
     * @brief Adds a memory field to the packer. Will manage the memory field free.
     * @param name The field name identifier.
     * @param ptr The memory pointer to manage. Cannot be manually freed after adding!
     * @param compression The compression applied to the field's stored bytes.
     * @param filter The filter applied to the field's elements before compression.
     * @throws RS_InvalidArgumentException if the memory pointer is provider-less.
     * @throws RS_InvalidArgumentException if the field name is reserved, exceeds the maximum
     * length, or has shape that exceeds the maximum dimensions, or if the field type is
     * unsupported.
     */
    void addField(
        const std::string_view& name,
        const RS_MemoryPtr& ptr,
        RSE_FastPackingCompressionMethod compression = RSE_FAST_PACKING_COMPRESSION_METHOD_NONE,
        RSE_FastPackingCompressionFilter filter = RSE_FAST_PACKING_COMPRESSION_FILTER_NONE
    ) {
        if (ptr.isWeak()) {
            RS_THROW(RS_InvalidArgumentException(
                "Cannot add a provider-less memory pointer to RS_FastPacker"
            ));
        }
        // Create RS_NdMemory<RByte> to manage the memory pointer
        // Size is ptr.size() since we're treating it as raw bytes
        addField(name, RS_NdMemory<RByte>(ptr, ptr.size(), true), compression, filter);
    }

    /**
     * @brief Adds a field to the packer.
     *
     * @tparam T The data type of the field.
     * @param name The name of the field.
     * @param field The field data.
     * @param compression The compression applied to the field's stored bytes.
     * @param filter The filter applied to the field's elements before compression.
     * @throws RS_InvalidArgumentException if the field name is reserved, exceeds the maximum
     * length, or has shape that exceeds the maximum dimensions, or if the field type is
     * unsupported.
     */
    template <typename T>
    void addField(
        const std::string_view& name,
        const RS_NdMemory<T>& field,
        RSE_FastPackingCompressionMethod compression = RSE_FAST_PACKING_COMPRESSION_METHOD_NONE,
        RSE_FastPackingCompressionFilter filter = RSE_FAST_PACKING_COMPRESSION_FILTER_NONE
    ) {
        if (name == RS_PACKING_META_FIELD_NAME) {
            RS_THROW(RS_InvalidArgumentException("Field name [{}] is reserved", name));
        }
        if (name.length() >= RS_PACKING_NAME_MAX_LEN) {
            RS_THROW(RS_InvalidArgumentException(
                "Field name [{}] exceeds maximum length {}", name, RS_PACKING_NAME_MAX_LEN - 1
            ));
        }
        if (field.shape.size() > RS_PACKING_SHAPE_MAX_LEN) {
            RS_THROW(RS_InvalidArgumentException(
                "Field with name [{}] has shape {} which exceeds maximum dimensions {}",
                name,
                fmt::join(field.shape, ","),
                RS_PACKING_SHAPE_MAX_LEN
            ));
        }
        RS_FastPackingFieldType type = RS_FastPackingFieldType::getType<T>();
        if (type.data_size == 0) {
            RS_THROW(
                RS_InvalidArgumentException("Field with name [{}] has unsupported type", name)
            );
        }
        RS_FastPackingHeaderRow field_header;
        memset(&field_header, 0, sizeof(field_header));
        std::memcpy(
            field_header.name,
            name.data(),
            std::min<std::size_t>(name.length(), RS_PACKING_NAME_MAX_LEN - 1)
        );
        field_header.type = type;
        field_header.dimensions = field.shape.size();
        RBigSize total_size = 1;
        for (auto&& [idx, dim_size] : enumerate(field.shape)) {
            field_header.shape[idx] = dim_size;
            total_size *= dim_size;
        }
        field_header.size = total_size;
        field_header.raw_size = total_size * sizeof(T);
        field_header.compression = compression;
        field_header.filter = filter;
        _fields.emplace_back(field.toMemoryPtr(), field_header);
        _buffers.emplace_back(field);
        _invalidateStored();
    }

    /**
     * @brief Clears the packer.
     */
    void clear() {
        _fields.clear();
        _buffers.clear();
        _invalidateStored();
    }

    /**
     * @brief Calculates the size of the packed data in bytes.
     *
     * @return The size of the packed data in bytes.
     */
    [[nodiscard]] RBigSize packSize() const {
        _ensureStored();
        RBigSize field_size = 0;
        for (auto& f : _fields) field_size += f.second.stored_size;
        return _getHeaderSize() + field_size;
    }

    /**
     * @brief Packs the data. The buffer must be larger or equal to the size
     * of packSize().
     *
     * @param dst Destination buffer.
     * @throws RS_InvalidArgumentException if the destination buffer is too small.
     */
    void pack(RS_MemoryPtr dst) const {
        if (dst.size() < packSize()) {
            RS_THROW(RS_InvalidArgumentException("Destination buffer is too small"));
        }
        auto it = getIterator();
        it.read(dst);
    }

    /**
     * @brief Packs the data into a file, prefixed with a caller-supplied magic.
     * @note The magic is a file-only concern, the in-memory blob has none. Never materializes
     * the whole blob.
     * @param path The destination file path.
     * @param magic The magic written at the start of the file.
     * @throws RS_FastPackingException if the file cannot be opened or written.
     */
    void pack(const std::string_view& path, uint32_t magic) const {
        _ensureStored();

        std::ofstream file(std::string(path), std::ios::binary | std::ios::trunc);
        if (!file) {
            RS_THROW(RS_FastPackingException("Cannot open file [{}] for writing", path));
        }

        uint32_t magic_wire = _is_byte_swap_needed ? serialization_helper::bswap32(magic) : magic;
        file.write(reinterpret_cast<const char*>(&magic_wire), sizeof(magic_wire));

        RBigSize header_size = _getHeaderSize();
        auto header = RS_NdMemory<RByte>(header_size, _provider, true);
        _writeHeader(header.toMemoryPtr());
        file.write(reinterpret_cast<const char*>(header.toMemoryPtr().as<char>()), header_size);

        for (auto&& [idx, f] : enumerate(_fields)) {
            RBigSize stored_size = f.second.stored_size;
            if (stored_size > 0) {
                file.write(
                    reinterpret_cast<const char*>(_stored_ptrs[idx].as<char>()), stored_size
                );
            }
        }

        if (!file) {
            RS_THROW(RS_FastPackingException("Failed writing file [{}]", path));
        }
    }

    /**
     * @brief Gets an iterator for streaming packing.
     *
     * @return An iterator for the packed data.
     */
    [[nodiscard]] RS_FastPackerIterator getIterator() const {
        _ensureStored();
        return RS_FastPackerIterator(this);
    }

   private:
    friend struct RS_FastPackerIterator;

    /// Memory provider for the packer.
    RS_MemoryProvider* _provider;

    /// Thread pool used to compress fields in parallel, or null for serial.
    RS_ThreadPool* _pool = nullptr;

    /// Whether the host requires byte swapping for the little endian wire format.
    bool _is_byte_swap_needed = false;

    /// Fields to be packed, each a source pointer and its header row.
    mutable std::vector<std::pair<RS_MemoryPtr, RS_FastPackingHeaderRow>> _fields;

    /// Source field memories, kept to maintain their existence during packing.
    std::vector<RS_Any> _buffers;

    /// Owned compressed bytes per field, parallel to _fields, empty when a field is stored as is.
    mutable std::vector<RS_NdMemory<RByte>> _compressed_buffers;

    /// Pointer to the stored bytes per field, aliasing the source or the compressed buffer.
    mutable std::vector<RS_MemoryPtr> _stored_ptrs;

    /// Whether the stored bytes cache is computed and valid.
    mutable bool _is_stored_ready = false;

    /**
     * @brief Invalidates the stored bytes cache.
     */
    void _invalidateStored() const {
        _compressed_buffers.clear();
        _stored_ptrs.clear();
        _is_stored_ready = false;
    }

    /**
     * @brief Calculates the size of the packing header in bytes.
     * @return The header size in bytes.
     */
    RBigSize _getHeaderSize() const {
        return sizeof(RS_FastPackingHeader) + sizeof(RS_FastPackingHeaderRow) * _fields.size();
    }

    /**
     * @brief Writes the packing header into a host buffer, byte-swapped when the host is not
     * little endian.
     * @param dst The host destination buffer, at least _getHeaderSize() bytes.
     */
    void _writeHeader(RS_MemoryPtr dst) const {
        auto h = dst.as<RS_FastPackingHeader>();
        memset(h, 0, sizeof(RS_FastPackingHeader));
        h->row_num = uint32_t(_fields.size());
        for (auto&& [idx, f] : enumerate(_fields)) h->rows[idx] = f.second;
        if (_is_byte_swap_needed) {
            for (RBigIndex row = 0; row < _fields.size(); row++) h->rows[row].byteSwap();
            h->row_num = serialization_helper::bswap32(h->row_num);
        }
    }

    /**
     * @brief Filters and compresses each field into its stored bytes, fixing stored_size and
     * offset.
     * @note Result is cached and reused until a mutation invalidates it.
     */
    void _ensureStored() const {
        if (_is_stored_ready) return;

        RBigSize field_num = _fields.size();
        _compressed_buffers.clear();
        _compressed_buffers.resize(field_num);
        _stored_ptrs.clear();
        _stored_ptrs.resize(field_num);

        if (_pool != nullptr && field_num > 0) {
            _pool->submitLoop(0, RIndex(field_num), [this](RIndex i) { _storeField(i); }).wait();
        } else {
            for (RIndex i = 0; i < field_num; i++) _storeField(i);
        }

        // Assign the byte offsets in field order.
        RBigSize offset = _getHeaderSize();
        for (auto& f : _fields) {
            f.second.offset = offset;
            offset += f.second.stored_size;
        }
        _is_stored_ready = true;
    }

    /**
     * @brief Filters and compresses a single field into its stored bytes, fixing its stored_size.
     * @param i The field index.
     */
    void _storeField(RIndex i) const {
        auto& row = _fields[i].second;
        RS_MemoryPtr src = _fields[i].first;

        // Store the source bytes directly when there is neither a method nor a filter.
        if (row.compression == RSE_FAST_PACKING_COMPRESSION_METHOD_NONE &&
            row.filter == RSE_FAST_PACKING_COMPRESSION_FILTER_NONE) {
            row.stored_size = row.raw_size;
            _stored_ptrs[i] = src;
            return;
        }

        RS_FastPackingCompressor compressor(row.compression, row.filter);
        RBigSize bound = compressor.getMaxCompressedSize(row.raw_size);
        auto compressed = RS_NdMemory<RByte>(bound, _provider, true);
        RBigSize stored_size =
            compressor.compress(compressed.toMemoryPtr(), src, row.type.data_size, row.size);
        row.stored_size = stored_size;
        _compressed_buffers[i] = compressed;
        _stored_ptrs[i] = RS_MemoryPtr(compressed.toMemoryPtr().as<RByte>(), stored_size);
    }
};

/**
 * @brief Class for unpacking data.
 */
class RS_FastUnpacker {
   public:
    /**
     * @brief Constructs a fast unpacker.
     * @param mem_provider The memory provider used for decompression staging.
     * @param pool The thread pool used to load fields in parallel, or null for serial.
     */
    explicit RS_FastUnpacker(
        RS_MemoryProvider* mem_provider = &RS_MemoryDefaultProvider::getInstance(),
        RS_ThreadPool* pool = nullptr
    ) {
        static_assert(
            isFastPackingValidFloatLayout(),
            "The platform must have the required IEEE 754 float layout"
        );
        _provider = mem_provider;
        _pool = pool;
        _is_byte_swap_needed = isFastPackingByteSwapNeeded();
    }

    /**
     * @brief Unpacks data from a complete buffer.
     * @note The buffer may be borrowed or mmap'd and provider-less when is_managed is false.
     * Uncompressed fields stay zero-copy slices into the buffer.
     * @param source Source buffer containing the complete packed data.
     * @param is_managed Whether to manage the memory of the source buffer.
     * @throws RS_FastPackingException if the source buffer is too small for the header or the
     * complete data.
     */
    void unpack(RS_MemoryPtr source, bool is_managed = false) {
        _reset();
        _is_file_backed = false;

        if (source.size() < sizeof(RS_FastPackingHeader)) {
            RS_THROW(RS_FastPackingException("Source buffer too small for header"));
        }
        _source = RS_NdMemory<RByte>(source, source.size(), is_managed);

        // Read the row count to learn the header size, then parse the whole header.
        const RByte* base = source.as<RByte>();
        uint32_t row_num_wire;
        memcpy(&row_num_wire, base, sizeof(uint32_t));
        uint32_t row_num = _is_byte_swap_needed ? serialization_helper::bswap32(row_num_wire)
                                                : row_num_wire;
        RBigSize header_size =
            sizeof(RS_FastPackingHeader) + RBigSize(row_num) * sizeof(RS_FastPackingHeaderRow);
        if (source.size() < header_size) {
            RS_THROW(RS_FastPackingException("Source buffer too small for complete header"));
        }
        _readHeader(base);

        // Slice the stored bytes for each field and validate the total size.
        _loaded_buffers.resize(_fields.size());
        for (auto& f : _fields) {
            RBigSize begin = f.second.offset;
            RBigSize end = begin + f.second.stored_size;
            if (source.size() < end) {
                RS_THROW(RS_FastPackingException("Source buffer too small for complete data"));
            }
            f.first = RS_MemoryPtr(source.as<RByte>() + begin, end - begin);
        }

        _is_complete = true;
    }

    /**
     * @brief Unpacks data from a file, verifying a caller-supplied magic.
     * @note When lazy, reads only the header and loads each field on demand, constant time
     * regardless of file size. When not lazy, reads and materializes every field up front.
     * @param path The source file path.
     * @param magic The magic expected at the start of the file.
     * @param is_lazy Whether to load fields on demand rather than all up front.
     * @throws RS_FastPackingException if the file cannot be opened, the magic does not match, or
     * the header cannot be read.
     */
    void unpack(const std::string_view& path, uint32_t magic, bool is_lazy = true) {
        _reset();
        _is_file_backed = true;

        _file.open(std::string(path), std::ios::binary);
        if (!_file) {
            RS_THROW(RS_FastPackingException("Cannot open file [{}] for reading", path));
        }

        uint32_t file_magic = 0;
        _file.read(reinterpret_cast<char*>(&file_magic), sizeof(file_magic));
        if (!_file) {
            RS_THROW(RS_FastPackingException("Failed reading magic from file [{}]", path));
        }
        if (_is_byte_swap_needed) file_magic = serialization_helper::bswap32(file_magic);
        if (file_magic != magic) {
            RS_THROW(RS_FastPackingException(
                "File [{}] magic mismatch: expected {}, got {}", path, magic, file_magic
            ));
        }

        // Read the row count, then the rows, into a contiguous header buffer.
        uint32_t row_num_wire = 0;
        _file.read(reinterpret_cast<char*>(&row_num_wire), sizeof(uint32_t));
        if (!_file) {
            RS_THROW(RS_FastPackingException("Failed reading header from file [{}]", path));
        }
        uint32_t row_num = _is_byte_swap_needed ? serialization_helper::bswap32(row_num_wire)
                                                : row_num_wire;
        RBigSize header_size =
            sizeof(RS_FastPackingHeader) + RBigSize(row_num) * sizeof(RS_FastPackingHeaderRow);

        auto header_buffer = std::make_unique<RByte[]>(header_size);
        memcpy(header_buffer.get(), &row_num_wire, sizeof(uint32_t));
        RBigSize rows_size = header_size - sizeof(RS_FastPackingHeader);
        if (rows_size > 0) {
            _file.read(
                reinterpret_cast<char*>(header_buffer.get() + sizeof(RS_FastPackingHeader)),
                std::streamsize(rows_size)
            );
            if (!_file) {
                RS_THROW(
                    RS_FastPackingException("Failed reading header rows from file [{}]", path)
                );
            }
        }
        _readHeader(header_buffer.get());
        _loaded_buffers.resize(_fields.size());

        // Eagerly materialize every field when not lazy. The reads stay serial because the single
        // file handle cannot be seeked concurrently.
        if (!is_lazy) {
            for (RBigIndex i = 0; i < _fields.size(); i++) _ensureLoaded(i);
        }

        _is_complete = true;
    }

    /**
     * @brief Retrieves the meta-data of the packed data.
     * @return The meta-data of the packed data.
     * @throws RS_FastPackingException if unpacking is not complete.
     */
    [[nodiscard]] RS_JSON getMetaData() const {
        if (!_is_complete) {
            RS_THROW(RS_FastPackingException("Unpacking not complete"));
        }
        if (_index.find(RS_PACKING_META_FIELD_NAME) == _index.end()) {
            return {};
        }
        auto meta_ptr = getField(RS_PACKING_META_FIELD_NAME);
        return RS_JSON::parse(meta_ptr.as<char>());
    }

    /**
     * @brief Checks if a field exists in the unpacked data.
     *
     * @param name Name of the field to check.
     * @return True if the field exists, false otherwise.
     * @throws RS_FastPackingException if unpacking is not complete.
     */
    [[nodiscard]] bool hasField(const std::string_view& name) const {
        if (!_is_complete) {
            RS_THROW(RS_FastPackingException("Unpacking not complete"));
        }
        return _index.find(std::string(name)) != _index.end();
    }

    /**
     * @brief Retrieves the list of field names in the unpacked data.
     *
     * @return A vector containing the names of all fields.
     */
    [[nodiscard]] std::vector<std::string> getFieldNames() const {
        if (!_is_complete) {
            RS_THROW(RS_FastPackingException("Unpacking not complete"));
        }

        std::vector<std::string> fields;
        fields.reserve(_index.size());
        for (const auto& [name, index] : _index) {
            fields.emplace_back(name);
        }
        return fields;
    }

    /**
     * @brief Retrieves the packing field type for a specified field name.
     *
     * @param name Name of the field whose type is to be retrieved.
     * @return The packing field type corresponding to the specified field name.
     * @throws RS_FastPackingException if unpacking is not complete.
     * @throws RS_NotFoundException if the field doesn't exist.
     */
    [[nodiscard]] RS_FastPackingFieldType getFieldType(const std::string_view& name) const {
        if (!_is_complete) {
            RS_THROW(RS_FastPackingException("Unpacking not complete"));
        }

        auto it = _index.find(std::string(name));
        if (it == _index.end()) {
            RS_THROW(RS_NotFoundException("Field '{}' does not exist", name));
        }

        return _fields[it->second].second.type;
    }

    /**
     * @brief Retrieves the raw memory pointer for a specified field.
     * @note Uncompressed and unfiltered fields on a memory backing are returned zero-copy,
     * otherwise the field is loaded and cached.
     * @param name The name of the field.
     * @return The raw memory pointer for the field.
     * @throws RS_NotFoundException if the field doesn't exist.
     */
    [[nodiscard]] RS_MemoryPtr getField(const std::string_view& name) const {
        if (!_is_complete) {
            RS_THROW(RS_FastPackingException("Unpacking not complete"));
        }
        auto it = _index.find(std::string(name));
        if (it == _index.end()) {
            RS_THROW(RS_NotFoundException("Field '{}' does not exist", name));
        }

        RBigIndex field_index = it->second;
        auto& row = _fields[field_index].second;
        if (!_is_file_backed && !_is_byte_swap_needed &&
            row.compression == RSE_FAST_PACKING_COMPRESSION_METHOD_NONE &&
            row.filter == RSE_FAST_PACKING_COMPRESSION_FILTER_NONE) {
            return _fields[field_index].first;
        }
        return _ensureLoaded(field_index);
    }

    /**
     * @brief Retrieves the data of a specified field as RS_NdMemory.
     *
     * @tparam T The data type of the field.
     * @param name Name of the field.
     * @return The field data as RS_NdMemory<T>.
     * @throws RS_NotFoundException if the field doesn't exist.
     * @throws RS_FastPackingException if the receiver type does not match the field type.
     */
    template <typename T>
    [[nodiscard]] RS_NdMemory<T> getField(const std::string_view& name) const {
        if (!_is_complete) {
            RS_THROW(RS_FastPackingException("Unpacking not complete"));
        }

        auto it = _index.find(std::string(name));
        if (it == _index.end()) {
            RS_THROW(RS_NotFoundException("Field '{}' does not exist", name));
        }

        RBigIndex field_index = it->second;
        auto& header_row = _fields[field_index].second;

        // Validate type compatibility
        if (RS_FastPackingFieldType::getType<T>() != header_row.type) {
            RS_THROW(RS_FastPackingException(
                "Field '{}' type mismatch: expected receiver to have type {}{}, got {}{}",
                name,
                header_row.type.data_type,
                header_row.type.data_size,
                RS_FastPackingFieldType::getType<T>().data_type,
                RS_FastPackingFieldType::getType<T>().data_size
            ));
        }

        RS_MemoryPtr field_ptr = getField(name);

        // Build shape vector
        std::vector<RBigSize> shape;
        shape.resize(header_row.dimensions);
        for (uint64_t dim = 0; dim < header_row.dimensions; dim++) {
            shape[dim] = header_row.shape[dim];
        }

        // Return RS_NdMemory that doesn't manage the memory
        return RS_NdMemory<T>(field_ptr, shape, false);
    }

    /**
     * @brief Checks if unpacking is complete.
     */
    [[nodiscard]] bool isComplete() const { return _is_complete; }

   private:
    /// Whether unpacking is complete.
    bool _is_complete = false;

    /// Memory provider used for decompression staging.
    RS_MemoryProvider* _provider;

    /// Thread pool used to load fields in parallel, or null for serial.
    RS_ThreadPool* _pool = nullptr;

    /// Whether the host requires byte swapping for the little endian wire format.
    bool _is_byte_swap_needed = false;

    /// Whether the source is a file, otherwise an in-memory buffer.
    bool _is_file_backed = false;

    /// In-memory source, wraps the passed buffer for the memory backing.
    RS_NdMemory<RByte> _source;

    /// Open file handle for the file backing.
    mutable std::ifstream _file;

    /// Size of the packing header in bytes.
    RBigSize _header_size = 0;

    /// Map containing field names and their indices.
    std::unordered_map<std::string, RBigIndex> _index;

    /// Fields, each a stored bytes pointer for the memory backing and its header row.
    std::vector<std::pair<RS_MemoryPtr, RS_FastPackingHeaderRow>> _fields;

    /// Owned loaded field bytes, parallel to _fields, filled lazily by _ensureLoaded.
    mutable std::vector<RS_NdMemory<RByte>> _loaded_buffers;

    /**
     * @brief Resets the unpacker to an empty state.
     */
    void _reset() {
        _is_complete = false;
        _is_file_backed = false;
        _source = RS_NdMemory<RByte>();
        _index.clear();
        _fields.clear();
        _loaded_buffers.clear();
        _header_size = 0;
        if (_file.is_open()) _file.close();
        _file.clear();
    }

    /**
     * @brief Parses the packing header from a contiguous header region into _fields and _index.
     * @param header_data Pointer to the header bytes, the row count followed by the rows.
     */
    void _readHeader(const RByte* header_data) {
        uint32_t row_num_wire;
        memcpy(&row_num_wire, header_data, sizeof(uint32_t));
        uint32_t row_num = _is_byte_swap_needed ? serialization_helper::bswap32(row_num_wire)
                                                : row_num_wire;

        const RByte* rows_data = header_data + sizeof(RS_FastPackingHeader);
        _fields.reserve(row_num);
        for (uint32_t r = 0; r < row_num; r++) {
            RS_FastPackingHeaderRow row;
            memcpy(&row, rows_data + RBigSize(r) * sizeof(RS_FastPackingHeaderRow), sizeof(row));
            if (_is_byte_swap_needed) row.byteSwap();
            _index.emplace(std::string(row.name), _fields.size());
            _fields.emplace_back(RS_MemoryPtr{}, row);
        }
        _header_size =
            sizeof(RS_FastPackingHeader) + RBigSize(row_num) * sizeof(RS_FastPackingHeaderRow);
    }

    /**
     * @brief Loads a field's bytes, decompressing and inverse filtering as needed, and caches it.
     * @param field_index The index of the field to load.
     * @return The pointer to the loaded raw bytes.
     */
    RS_MemoryPtr _ensureLoaded(RBigIndex field_index) const {
        if (!_loaded_buffers[field_index].empty()) {
            return _loaded_buffers[field_index].toMemoryPtr();
        }

        auto& row = _fields[field_index].second;

        // Obtain the stored bytes, from the file for a file backing or the source slice otherwise.
        RS_NdMemory<RByte> stored_owned;
        RS_MemoryPtr stored;
        if (_is_file_backed) {
            stored_owned = RS_NdMemory<RByte>(row.stored_size, _provider, true);
            stored = stored_owned.toMemoryPtr();
            _readFieldBytes(row, stored);
        } else {
            stored = _fields[field_index].first;
        }

        // Decompress and inverse filter into one raw-size buffer, or copy when both are none.
        auto raw = RS_NdMemory<RByte>(row.raw_size, _provider, true);
        RS_MemoryPtr raw_ptr = raw.toMemoryPtr();
        if (row.compression == RSE_FAST_PACKING_COMPRESSION_METHOD_NONE &&
            row.filter == RSE_FAST_PACKING_COMPRESSION_FILTER_NONE) {
            raw_ptr.copyFrom(stored, row.raw_size);
        } else {
            RS_FastPackingCompressor compressor(row.compression, row.filter);
            compressor.decompress(
                raw_ptr, stored, row.stored_size, row.raw_size, row.type.data_size, row.size
            );
        }

        if (_is_byte_swap_needed) {
            byteSwapFastPackingElements(raw_ptr, row.size, row.type.data_size);
        }

        _loaded_buffers[field_index] = raw;
        return raw.toMemoryPtr();
    }

    /**
     * @brief Reads a field's stored bytes from the file into a destination buffer.
     * @param row The header row of the field.
     * @param dst The destination buffer, at least row.stored_size bytes.
     */
    void _readFieldBytes(const RS_FastPackingHeaderRow& row, RS_MemoryPtr dst) const {
        _file.clear();
        _file.seekg(std::streamoff(sizeof(uint32_t) + row.offset), std::ios::beg);
        _file.read(reinterpret_cast<char*>(dst.as<char>()), std::streamsize(row.stored_size));
        if (!_file) {
            RS_THROW(
                RS_FastPackingException("Failed reading field '{}' bytes from file", row.name)
            );
        }
    }
};

}  // namespace rise

#endif  // RS_FAST_PACK_H
