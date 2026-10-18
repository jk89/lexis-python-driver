# SPDX-License-Identifier: Apache-2.0

"""
Lexis' order-preserving partition key encoding.

Lexis partitions tables with ``org.lexis.dht.LexisByteOrderedPartitioner``:
the token of a partition is an order-preserving encoding of its partition key,
so token order is partition key order. This module computes that encoding
(it must match the server's ``dht/ordered_encoding.cc`` byte for byte), and
the 64-bit projection the server uses for shard routing.

Each component is type-encoded, its 0x00 bytes are escaped as 0x00 0x01, and
it is terminated by 0x00 0x00. A component of a column declared DESC in
``PARTITION ORDER BY`` is that encoding bitwise complemented, terminator
included. The design (``Direction``, the 0xFF 0xFF descending terminator) is
from kaebase:
https://github.com/kaepek/kaebase/blob/FEAT/scaling/v2/crates/common/src/encoding/cluster_key.rs
"""

import itertools
import math
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
    # CQL compares floats as Java does: every NaN equals every other NaN and
    # sorts above everything, so all NaNs encode as the canonical quiet NaN.
    if math.isnan(v):
        v = float('nan')
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


def _time_fields(raw):
    # time_hi (version masked), time_mid, time_low.
    return bytes([raw[6] & 0x0f, raw[7], raw[4], raw[5], raw[0], raw[1], raw[2], raw[3]])


def _encode_timeuuid(raw):
    # CQL order: the time fields, the clock sequence and node as signed bytes,
    # then the version nibble.
    return _time_fields(raw) + bytes(b ^ 0x80 for b in raw[8:]) + bytes([raw[6] >> 4])


def _encode_uuid(raw):
    # CQL order: the version, then for version 1 the time fields and the rest
    # unsigned, for other versions all bytes in order.
    version = raw[6] >> 4
    if version == 1:
        return bytes([version]) + _time_fields(raw) + raw[8:]
    return bytes([version]) + raw


class Descending(object):
    """A partition key column type declared DESC in ``PARTITION ORDER BY``."""

    __slots__ = ('type',)

    def __init__(self, cql_type):
        self.type = cql_type

    def __repr__(self):
        return 'Descending(%r)' % (self.type,)


def _direction(cql_type):
    """(type, descending) of a partition key column type: a Descending, a
    ``desc(<type>)`` string, or a plain type."""
    if isinstance(cql_type, Descending):
        return cql_type.type, True
    if isinstance(cql_type, str):
        s = cql_type.strip()
        if s.lower().startswith('desc(') and s.endswith(')'):
            return s[5:-1], True
    return cql_type, False


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
    elif type_name == 'timeuuid' and n == 16:
        _escape_into(out, _encode_timeuuid(raw))
    elif type_name == 'uuid' and n == 16:
        _escape_into(out, _encode_uuid(raw))
    else:
        # Text, ascii, blob, inet, duration, and types with no
        # order-preserving rule (tuples, UDTs, collections): raw bytes.
        _escape_into(out, raw)
    out += b'\x00\x00'


def encode(types, components):
    """
    Encodes the serialized components of a partition key. `types` are the
    partition key column types (CQL type names or cqltypes classes), each
    wrapped in :class:`Descending` if the column is DESC.
    """
    directions = [_direction(t) for t in types]
    out = bytearray()
    for i, raw in enumerate(components):
        raw = bytes(raw)
        if i >= len(directions):
            # More components than partition key columns: raw bytes, like the server.
            _escape_into(out, raw)
            out += b'\x00\x00'
            continue
        cql_type, descending = directions[i]
        mark = len(out)
        _encode_component(out, _base_type_name(cql_type), raw)
        if descending:
            for j in range(mark, len(out)):
                out[j] ^= 0xff
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


def successor(p):
    """
    The smallest byte string greater than every string that starts with p
    (trailing 0xff bytes dropped, then the last byte incremented), or None if
    p is all 0xff. The server's dht::ordered_encoding::prefix_successor().
    """
    p = bytes(p).rstrip(b'\xff')
    if not p:
        return None
    return p[:-1] + bytes([p[-1] + 1])


ROUTING_PLAN_PAYLOAD = 'lexis-prefix-routing-v1'


def routing_token(query):
    """The routing plan token (bytes) of a statement, or None."""
    t = getattr(query, 'lexis_routing_token', None)
    return bytes(t) if isinstance(t, (bytes, bytearray)) else None
_ROUTING_PLAN_TYPE = ('ListType(TupleType(UTF8Type, BooleanType, ByteType, '
                      'ListType(TupleType(Int32Type, BytesType, ByteType))))')
_EQ, _IN, _IN_LIST, _SLICE, _TOKEN = range(5)
_UPPER, _INCLUSIVE = 1, 2


def _list_elements(serialized):
    """The elements of a serialized CQL list, or None if one is null."""
    count, = struct.unpack_from('>i', serialized, 0)
    pos = 4
    elements = []
    for _ in range(count):
        length, = struct.unpack_from('>i', serialized, pos)
        pos += 4
        if length < 0:
            return None
        elements.append(serialized[pos:pos + length])
        pos += length
    return elements


class RoutingPlan(object):
    """
    How to compute, from a bound statement's values, the token its partition
    ranges start at: the lexis-prefix-routing-v1 payload of a PREPARED
    response (LEXIS_PREFIX_ROUTING_V1). It lets prefix and range reads
    (``WHERE tenant = ?``, ``WHERE tenant = ? AND user > ?``, token() ranges)
    be routed, and carries the partition key directions, so complete-key reads
    of DESC tables are routed without schema metadata. See the server's
    statement_restrictions::lexis_routing_plan().
    """

    def __init__(self, entries):
        # [(cql type name, descending, operation, [(marker, literal, flags)])]
        self.entries = entries

    @classmethod
    def from_payload(cls, data, protocol_version):
        from cassandra.cqltypes import lookup_casstype
        entries = lookup_casstype(_ROUTING_PLAN_TYPE).from_binary(data, protocol_version)
        return cls([(t, desc, op, list(values)) for t, desc, op, values in entries])

    def start_token(self, values):
        """The token bytes the read starts at, or None if it cannot be computed."""
        def value(v):
            marker, literal, _ = v
            if marker < 0:
                return literal
            if marker >= len(values) or not isinstance(values[marker], (bytes, bytearray)):
                return None
            return bytes(values[marker])

        if self.entries and self.entries[0][2] == _TOKEN:
            start = b''
            for v in self.entries[0][3]:
                if v[2] & _UPPER:
                    continue
                b = value(v)
                if b is None:
                    return None
                start = b if v[2] & _INCLUSIVE else b + b'\x00'
            return start

        types, choices, range_column = [], [], None
        for cql_type, desc, op, vals in self.entries:
            typ = Descending(cql_type) if desc else cql_type
            if op == _SLICE:
                range_column = (typ, desc, vals)
                continue
            if op == _EQ or op == _IN:
                options = [value(v) for v in vals]
            elif op == _IN_LIST:
                serialized = value(vals[0])
                options = _list_elements(serialized) if serialized is not None else None
            else:
                return None
            if options is None or any(o is None for o in options):
                return None
            types.append(typ)
            choices.append(options)

        best = None
        # As get_pk_ordered_prefix_ranges(): one range per prefix combination.
        for combo in itertools.product(*choices):
            p = encode(types, combo) if combo else None
            lo, hi = p, (successor(p) if p is not None else None)
            skip = False
            if range_column is not None:
                typ, desc, vals = range_column
                for v in vals:
                    b = value(v)
                    if b is None:
                        return None
                    e = encode(types + [typ], list(combo) + [b])
                    inclusive = bool(v[2] & _INCLUSIVE)
                    # A DESC column encodes in the opposite order: its CQL upper
                    # bound is the low end of the range.
                    if bool(v[2] & _UPPER) == desc:
                        lo = e if inclusive else successor(e)
                        if lo is None:
                            skip = True
                    else:
                        hi = successor(e) if inclusive else e
            if skip or (lo is not None and hi is not None and hi <= lo):
                continue
            start = lo if lo is not None else b''
            if best is None or start < best:
                best = start
        return best
