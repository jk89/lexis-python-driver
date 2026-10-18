# SPDX-License-Identifier: Apache-2.0

import os
import struct
import unittest
import uuid
from unittest.mock import Mock

from cassandra import lexis
from cassandra.cqltypes import Int32Type, UTF8Type
from cassandra.metadata import (Metadata, KeyspaceMetadata, TableMetadata, ColumnMetadata,
                                LexisToken)
from cassandra.policies import SimpleConvictionPolicy
from cassandra.pool import Host
from cassandra.protocol_features import ProtocolFeatures
from cassandra.shard_info import _ShardingInfo
from cassandra.tablets import Tablet, Tablets


def _composite(parts):
    return b"".join(struct.pack(">H%dsB" % len(p), len(p), p, 0) for p in parts)


class LexisEncodingTest(unittest.TestCase):

    def test_matches_server_vectors(self):
        # "<types> <serialized components> <token>", generated from the server's
        # encoder (dht/ordered_encoding.cc). DESC columns are written desc(<type>).
        path = os.path.join(os.path.dirname(__file__), "lexis_ordered_encoding_vectors.txt")
        n = 0
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                types, components, token = line.split(" ")
                got = lexis.encode(types.split(","), [bytes.fromhex(c) for c in components.split(",")])
                self.assertEqual(got.hex(), token, line)
                n += 1
        self.assertEqual(n, 1030)

    def test_tokens_are_key_ordered(self):
        values = [-2 ** 31, -70000, -1, 0, 1, 255, 256, 2 ** 31 - 1]
        tokens = [LexisToken.from_key(struct.pack(">i", v), ["int"]) for v in values]
        self.assertEqual(tokens, sorted(tokens))
        self.assertEqual(LexisToken.from_key(struct.pack(">i", 143), ["int"]).value.hex(), "80000100018f0000")

    def test_composite_routing_key(self):
        key = _composite([b"b", struct.pack(">i", 5)])
        token = LexisToken.from_key(key, ["text", "int"])
        self.assertEqual(token.value, lexis.encode(["text", "int"], [b"b", struct.pack(">i", 5)]))
        # cqltypes classes (prepared statement metadata) work like CQL type names.
        self.assertEqual(token, LexisToken.from_key(key, [UTF8Type, Int32Type]))
        self.assertRaises(ValueError, LexisToken.from_key, key)

    def test_desc_is_complement(self):
        asc = lexis.encode(["text"], [b"xyz"])
        desc = lexis.encode([lexis.Descending("text")], [b"xyz"])
        self.assertEqual(desc, bytes(b ^ 0xff for b in asc))
        self.assertEqual(desc, lexis.encode(["desc(text)"], [b"xyz"]))
        # "xyz" > "xy", so DESC puts "xyz" first.
        self.assertLess(desc, lexis.encode([lexis.Descending("text")], [b"xy"]))

    def test_mixed_directions(self):
        types = ["text", lexis.Descending("int")]
        rows = [(b"a", 5), (b"a", 1), (b"a", -1), (b"b", 7), (b"b", 0)]
        tokens = [lexis.encode(types, [t, struct.pack(">i", v)]) for t, v in rows]
        self.assertEqual(tokens, sorted(tokens))

    def test_from_string_is_hex(self):
        self.assertEqual(LexisToken.from_string("80000100018f0000").value, bytes.fromhex("80000100018f0000"))

    def test_shard_matches_server_projection(self):
        for shards, ignore_msb in ((1, 0), (7, 0), (8, 12), (14, 12)):
            info = _ShardingInfo(0, shards, "org.lexis.dht.LexisByteOrderedPartitioner", "biased-token-round-robin",
                                 ignore_msb, None, None)
            for v in (-5, 0, 143, 70000, 2 ** 31 - 1):
                token = LexisToken.from_key(struct.pack(">i", v), ["int"])
                # dht::zero_based_shard_of(unbias(t)), unbias(t) = first 8 bytes, zero-padded.
                unbiased = int.from_bytes(token.value[:8].ljust(8, b"\x00"), "big")
                expected = (((unbiased << ignore_msb) & (2 ** 64 - 1)) * shards) >> 64
                self.assertEqual(info.shard_id_from_token(token.shard_value), expected)


def _plan(entries):
    """A lexis-prefix-routing-v1 payload: [(type, desc, op, [(marker, literal, flags)])]."""
    from cassandra.cqltypes import lookup_casstype
    data = lookup_casstype(lexis._ROUTING_PLAN_TYPE).to_binary(entries, 4)
    return lexis.RoutingPlan.from_payload(data, 4)


def _int(v):
    return struct.pack(">i", v)


def _list(*elements):
    return struct.pack(">i", len(elements)) + b"".join(struct.pack(">i", len(e)) + e for e in elements)


class LexisRoutingPlanTest(unittest.TestCase):
    # Operations and flags, as the server's statement_restrictions::lexis_routing_plan().
    EQ, IN, IN_LIST, SLICE, TOKEN = range(5)
    UPPER, INCLUSIVE = 1, 2

    def test_successor(self):
        self.assertEqual(lexis.successor(b"a\x00\x00"), b"a\x00\x01")
        self.assertEqual(lexis.successor(b"a\xff\xff"), b"b")
        self.assertIsNone(lexis.successor(b"\xff\xff"))

    def test_prefix_eq(self):
        # tenant = ?  starts at the encoded prefix.
        plan = _plan([("text", False, self.EQ, [(0, b"", 0)])])
        self.assertEqual(plan.start_token([b"b"]), lexis.encode(["text"], [b"b"]))
        # A literal prefix value needs no bound value.
        plan = _plan([("text", False, self.EQ, [(-1, b"b", 0)])])
        self.assertEqual(plan.start_token([]), lexis.encode(["text"], [b"b"]))

    def test_complete_key_desc(self):
        # Complete keys get their token, with the directions from the plan.
        plan = _plan([("text", False, self.EQ, [(0, b"", 0)]), ("int", True, self.EQ, [(1, b"", 0)])])
        self.assertEqual(plan.start_token([b"b", _int(5)]),
                         lexis.encode(["text", lexis.Descending("int")], [b"b", _int(5)]))

    def test_range_ascending(self):
        prefix = ("text", False, self.EQ, [(0, b"", 0)])
        gt = _plan([prefix, ("int", False, self.SLICE, [(1, b"", 0)])])
        gte = _plan([prefix, ("int", False, self.SLICE, [(1, b"", self.INCLUSIVE)])])
        lt = _plan([prefix, ("int", False, self.SLICE, [(1, b"", self.UPPER)])])
        e = lexis.encode(["text", "int"], [b"b", _int(5)])
        self.assertEqual(gte.start_token([b"b", _int(5)]), e)
        self.assertEqual(gt.start_token([b"b", _int(5)]), lexis.successor(e))
        # Only an upper bound: the range starts at the prefix.
        self.assertEqual(lt.start_token([b"b", _int(5)]), lexis.encode(["text"], [b"b"]))

    def test_range_descending_swaps_bounds(self):
        # On a DESC column the CQL upper bound is the low end of the range.
        prefix = ("text", False, self.EQ, [(0, b"", 0)])
        lt = _plan([prefix, ("int", True, self.SLICE, [(1, b"", self.UPPER)])])
        gt = _plan([prefix, ("int", True, self.SLICE, [(1, b"", 0)])])
        two_sided = _plan([prefix, ("int", True, self.SLICE, [(1, b"", self.INCLUSIVE), (2, b"", self.UPPER | self.INCLUSIVE)])])
        types = ["text", lexis.Descending("int")]
        self.assertEqual(lt.start_token([b"b", _int(5)]), lexis.successor(lexis.encode(types, [b"b", _int(5)])))
        self.assertEqual(gt.start_token([b"b", _int(5)]), lexis.encode(["text"], [b"b"]))
        self.assertEqual(two_sided.start_token([b"b", _int(1), _int(9)]), lexis.encode(types, [b"b", _int(9)]))
        # An empty range (lower bound above the upper one) has no start.
        self.assertIsNone(two_sided.start_token([b"b", _int(9), _int(1)]))

    def test_in_starts_at_smallest(self):
        plan = _plan([("text", False, self.IN, [(0, b"", 0), (1, b"", 0)])])
        self.assertEqual(plan.start_token([b"z", b"c"]), lexis.encode(["text"], [b"c"]))
        plan = _plan([("text", True, self.IN_LIST, [(0, b"", 0)])])
        # DESC: "z" is the smallest token.
        self.assertEqual(plan.start_token([_list(b"c", b"z")]), lexis.encode([lexis.Descending("text")], [b"z"]))

    def test_token_range(self):
        plan = _plan([("blob", False, self.TOKEN, [(0, b"", self.INCLUSIVE), (1, b"", self.UPPER)])])
        self.assertEqual(plan.start_token([b"\x80", b"\x90"]), b"\x80")
        plan = _plan([("blob", False, self.TOKEN, [(0, b"", 0)])])
        self.assertEqual(plan.start_token([b"\x80"]), b"\x80\x00")
        plan = _plan([("blob", False, self.TOKEN, [(0, b"", self.UPPER)])])
        self.assertEqual(plan.start_token([b"\x80"]), b"")

    def test_unbound_value_gives_no_token(self):
        plan = _plan([("text", False, self.EQ, [(0, b"", 0)])])
        self.assertIsNone(plan.start_token([None]))
        self.assertIsNone(plan.start_token([]))

    def test_metadata_routes_by_plan_token(self):
        metadata = Metadata()
        metadata.rebuild_token_map("org.lexis.dht.LexisByteOrderedPartitioner",
                                   {Host("127.0.0.1", SimpleConvictionPolicy): ["ff"]})
        query = Mock(lexis_routing_token=b"\x01\x02")
        self.assertEqual(metadata.token_for_key("ks", "t", None, query), LexisToken(b"\x01\x02"))
        self.assertIsNone(metadata.token_for_key("ks", "t", None, Mock(lexis_routing_token=None)))


class LexisMetadataTest(unittest.TestCase):

    def _metadata(self):
        metadata = Metadata()
        ks = KeyspaceMetadata("ks", True, "SimpleStrategy", {"replication_factor": "1"})
        table = TableMetadata("ks", "t")
        table.partition_key = [ColumnMetadata(table, "tenant", "text"), ColumnMetadata(table, "user", "int")]
        ks.tables["t"] = table
        metadata.keyspaces["ks"] = ks
        self.hosts = [Host("127.0.0.%d" % i, SimpleConvictionPolicy) for i in (1, 2)]
        split = lexis.encode(["text", "int"], [b"m", struct.pack(">i", 0)])
        metadata.rebuild_token_map("org.lexis.dht.LexisByteOrderedPartitioner",
                                   {self.hosts[0]: [split.hex()], self.hosts[1]: ["ff"]})
        return metadata

    def test_token_map_and_replicas(self):
        metadata = self._metadata()
        self.assertIs(metadata.token_map.token_class, LexisToken)
        low = _composite([b"b", struct.pack(">i", 5)])
        high = _composite([b"x", struct.pack(">i", 5)])
        self.assertEqual(metadata.get_replicas("ks", low, table="t"), [self.hosts[0]])
        self.assertEqual(metadata.get_replicas("ks", high, table="t"), [self.hosts[1]])
        # Without the table the key types are unknown: no token, no replicas.
        self.assertEqual(metadata.get_replicas("ks", low), [])
        self.assertIsNone(metadata.token_for_key("ks", "nope", low))

    def test_token_from_prepared_statement_types(self):
        # Without schema metadata for the table, the key types come from the
        # bound statement's prepared metadata.
        metadata = self._metadata()
        prepared = Mock(routing_key_indexes=[0, 1],
                        column_metadata=[Mock(type=UTF8Type), Mock(type=Int32Type)])
        query = Mock(prepared_statement=prepared)
        key = _composite([b"b", struct.pack(">i", 5)])
        self.assertEqual(metadata.token_for_key("ks", "unknown_table", key, query),
                         LexisToken.from_key(key, ["text", "int"]))


    def test_desc_partition_key_from_schema(self):
        # Direction comes from schema metadata (clustering_order DESC on a
        # partition key column); prepared metadata does not carry it.
        metadata = self._metadata()
        table = metadata.keyspaces["ks"].tables["t"]
        table.partition_key[1].is_reversed = True
        key = _composite([b"b", struct.pack(">i", 5)])
        self.assertEqual(metadata.token_for_key("ks", "t", key),
                         LexisToken.from_key(key, ["text", lexis.Descending("int")]))
        self.assertIn('PARTITION ORDER BY (tenant ASC, user DESC)', table.as_cql_query())


class LexisTabletsTest(unittest.TestCase):

    def test_lookup_with_tagged_boundaries(self):
        split = LexisToken.from_key(struct.pack(">i", 0), ["int"])
        a, b = uuid.uuid4(), uuid.uuid4()
        tablets = Tablets({})
        tablets.add_tablet("ks", "t", Tablet.from_row(b"\x00", split.tablet_value, [(a, 0)]))
        tablets.add_tablet("ks", "t", Tablet.from_row(split.tablet_value, b"\x02", [(b, 1)]))

        def replicas(v):
            return tablets.get_tablet_for_key("ks", "t", LexisToken.from_key(struct.pack(">i", v), ["int"])).replicas

        self.assertEqual(replicas(-1), [(a, 0)])
        self.assertEqual(replicas(0), [(a, 0)])  # (first, last]
        self.assertEqual(replicas(1), [(b, 1)])
        self.assertEqual(replicas(2 ** 31 - 1), [(b, 1)])

    def test_negotiation(self):
        features = ProtocolFeatures.parse_from_supported({"LEXIS_TABLETS_ROUTING_V1": [""]})
        self.assertTrue(features.tablets_routing_v1)
        options = {}
        features.add_startup_options(options)
        self.assertEqual(options, {"LEXIS_TABLETS_ROUTING_V1": ""})

        features = ProtocolFeatures.parse_from_supported({"LEXIS_TABLETS_ROUTING_V1": [""], "LEXIS_PREFIX_ROUTING_V1": [""]})
        self.assertTrue(features.prefix_routing_v1)
        options = {}
        features.add_startup_options(options)
        self.assertEqual(options, {"LEXIS_TABLETS_ROUTING_V1": "", "LEXIS_PREFIX_ROUTING_V1": ""})

        features = ProtocolFeatures.parse_from_supported({"TABLETS_ROUTING_V1": [""]})
        options = {}
        features.add_startup_options(options)
        self.assertEqual(options, {"TABLETS_ROUTING_V1": ""})

        self.assertFalse(ProtocolFeatures.parse_from_supported({}).tablets_routing_v1)
