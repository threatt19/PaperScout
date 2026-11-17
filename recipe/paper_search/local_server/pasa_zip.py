"""Tolerant reader for PaSa ZIP entries with mismatched Unicode local names."""

from __future__ import annotations

import binascii
import struct
import zipfile


_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_LOCAL_SIGNATURE = b"PK\x03\x04"


def _read_without_name_check(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Read by central-directory offsets, ignoring only the local filename mismatch.

    PaSa's archive contains valid payloads whose local-header Unicode filenames differ
    from their central-directory filenames. Python's normal ZipFile reader rejects
    those entries before checking the payload. Sizes, compression type and CRC still
    come from the central directory and are verified here.
    """
    if archive.fp is None:
        raise ValueError("ZIP archive is closed")
    if info.flag_bits & 0x1:
        raise RuntimeError("Encrypted ZIP entries are not supported")
    with archive._lock:  # type: ignore[attr-defined]
        archive.fp.seek(info.header_offset)
        raw_header = archive.fp.read(_LOCAL_HEADER.size)
        if len(raw_header) != _LOCAL_HEADER.size:
            raise zipfile.BadZipFile("Truncated local file header")
        fields = _LOCAL_HEADER.unpack(raw_header)
        if fields[0] != _LOCAL_SIGNATURE:
            raise zipfile.BadZipFile("Bad local file header signature")
        filename_length = fields[-2]
        extra_length = fields[-1]
        archive.fp.seek(filename_length + extra_length, 1)
        compressed = archive.fp.read(info.compress_size)
    if len(compressed) != info.compress_size:
        raise zipfile.BadZipFile("Truncated compressed payload")

    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        decompressor = zipfile._get_decompressor(info.compress_type)  # type: ignore[attr-defined]
        if decompressor is None:
            data = compressed
        else:
            data = decompressor.decompress(compressed)
            flush = getattr(decompressor, "flush", None)
            if flush is not None:
                data += flush()

    if len(data) != info.file_size:
        raise zipfile.BadZipFile(
            f"Uncompressed size mismatch for {info.filename!r}: {len(data)} != {info.file_size}"
        )
    crc = binascii.crc32(data) & 0xFFFFFFFF
    if crc != info.CRC:
        raise zipfile.BadZipFile(f"CRC mismatch for {info.filename!r}")
    return data


def read_zip_entry(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    try:
        return archive.read(info)
    except zipfile.BadZipFile as exc:
        if "File name in directory" not in str(exc):
            raise
        return _read_without_name_check(archive, info)
