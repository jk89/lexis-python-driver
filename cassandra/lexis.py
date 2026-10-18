# SPDX-License-Identifier: Apache-2.0

"""
Lexis' order-preserving partition key encoding.

Lexis partitions tables with ``org.lexis.dht.LexisByteOrderedPartitioner``:
the token of a partition is an order-preserving encoding of its partition key,
so token order is partition key order. This module computes that encoding
(it must match the server's ``dht/ordered_encoding.cc`` byte for byte), and
the 64-bit projection the server uses for shard routing.

Each component is type-encoded, its 0x00 bytes are escaped as 0x00 0x01, and
it is terminated by 0x00 0x00.
"""

import struct

INT64_MIN = -(2 ** 63)
_U64 = (1 << 64) - 1

# Partition key column types with an order-preserving rule, by CQL type name.
_LONG = ('bigint', 'counter')
_TIMES = ('timestamp', 'date', 'time')


def _escape_into(out, data):
    for b in data:
        if b == 0:
            out += b'\x00\x01'
        else:
            out.append(b)


# Float lex transform ported from kaebase v2:
# https://github.com/kaepek/kaebase/blob/FEAT/scaling/v2/crates/common/src/encoding/float.rs
def _float64_to_lex(v):
    bits, = struct.unpack('>Q', struct.pack('>d', v))
    if bits & (1 << 63):
        return ~bits & _U64
    return bits | (1 << 63)


def _magnitude(twos):
    """Magnitude (big-endian, no leading zeros) and sign of a two's complement integer."""
    n = int.from_bytes(twos, 'big', signed=True) if twos else 0
    negative = n < 0
    n = abs(n)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big'), negative


def _encode_varint(out, raw):
    mag, negative = _magnitude(raw)
    flip = 0xff if negative else 0x00
    head = bytearray([0x7f if negative else 0x80])
    head += struct.pack('>I', len(mag) ^ (0xffffffff if negative else 0))
    head += bytes(b ^ flip for b in mag)
    _escape_into(out, head)


def _encode_decimal(out, raw):
    if len(raw) < 4:
        return False
    scale, = struct.unpack('>i', raw[:4])
    mag, negative = _magnitude(raw[4:])
    if not mag:
        _escape_into(out, b'\x80')
        return True
    digits = str(int.from_bytes(mag, 'big'))
    stripped = digits.rstrip('0') or '0'
    scale -= len(digits) - len(stripped)
    digits = stripped
    exponent = len(digits) - scale
    biased = (exponent ^ INT64_MIN) & _U64
    flip = 0xff if negative else 0x00
    enc = bytearray([0x7f if negative else 0x81])
    enc += struct.pack('>Q', (~biased & _U64) if negative else biased)
    enc += bytes(ord(c) ^ flip for c in digits)
    if negative:
        enc.append(0xff)
    _escape_into(out, enc)
    return True


def _base_type_name(cql_type):
    """CQL type name of a partition key column, from a type string or a cqltypes class."""
    if not isinstance(cql_type, str):
        cql_type = getattr(cql_type, 'typename', '') or ''
    name = cql_type.strip().lower()
    if name == 'varchar':
        name = 'text'
    return name


def _encode_component(out, type_name, raw):
    n = len(raw)
    if n == 0:
        out += b'\x00\x00'
        return
    if type_name in _LONG and n == 8:
        v, = struct.unpack('>q', raw)
        _escape_into(out, struct.pack('>Q', (v ^ INT64_MIN) & _U64))
    elif type_name == 'int' and n == 4:
        v, = struct.unpack('>I', raw)
        _escape_into(out, struct.pack('>I', v ^ 0x80000000))
    elif type_name == 'smallint' and n == 2:
        v, = struct.unpack('>H', raw)
        _escape_into(out, struct.pack('>H', v ^ 0x8000))
    elif type_name == 'tinyint' and n == 1:
        _escape_into(out, bytes([raw[0] ^ 0x80]))
    elif type_name == 'boolean' and n == 1:
        _escape_into(out, b'\x01' if raw[0] != 0 else b'\x00')
    elif type_name in _TIMES:
        # Up to 8 bytes read as an unsigned big-endian number, plus 2^63.
        v = int.from_bytes(raw[:8], 'big')
        _escape_into(out, struct.pack('>Q', (v + (1 << 63)) & _U64))
    elif type_name == 'double' and n == 8:
        v, = struct.unpack('>d', raw)
        _escape_into(out, struct.pack('>Q', _float64_to_lex(v)))
    elif type_name == 'float' and n == 4:
        v, = struct.unpack('>f', raw)
        _escape_into(out, struct.pack('>Q', _float64_to_lex(v)))
    elif type_name == 'varint':
        _encode_varint(out, raw)
    elif type_name == 'decimal':
        if not _encode_decimal(out, raw):
            _escape_into(out, raw)
    else:
        # Text, ascii, blob, uuid, timeuuid, inet, duration, and types with no
        # order-preserving rule (tuples, UDTs, collections): raw bytes.
        _escape_into(out, raw)
    out += b'\x00\x00'


def encode(types, components):
    """
    Encodes the serialized components of a partition key. `types` are the
    partition key column types (CQL type names or cqltypes classes).
    """
    names = [_base_type_name(t) for t in types]
    out = bytearray()
    for i, raw in enumerate(components):
        raw = bytes(raw)
        if i >= len(names):
            # More components than partition key columns: raw bytes, like the server.
            _escape_into(out, raw)
            out += b'\x00\x00'
        else:
            _encode_component(out, names[i], raw)
    return bytes(out)


def split_routing_key(routing_key, component_count):
    """
    Splits a driver routing key into its serialized components. A routing key
    of a single-column partition key is the column value itself; otherwise it
    is the legacy composite form (2-byte length, value, 0x00 per component).
    """
    if component_count <= 1:
        return [bytes(routing_key)]
    parts = []
    pos = 0
    key = bytes(routing_key)
    while pos < len(key):
        length, = struct.unpack_from('>H', key, pos)
        pos += 2
        parts.append(key[pos:pos + length])
        pos += length + 1
    return parts


def projection(token_bytes):
    """
    The server's 64-bit projection of a token (dht::token::unbias(): the first
    8 bytes, big-endian, zero-padded), in the signed form the driver's shard
    computation takes for Murmur3 tokens.
    """
    unbiased = int.from_bytes(token_bytes[:8].ljust(8, b'\x00'), 'big')
    return unbiased + INT64_MIN
