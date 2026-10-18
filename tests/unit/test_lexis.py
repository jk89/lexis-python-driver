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
        # encoder (dht/ordered_encoding.cc).
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
        self.assertEqual(n, 640)

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

        features = ProtocolFeatures.parse_from_supported({"TABLETS_ROUTING_V1": [""]})
        options = {}
        features.add_startup_options(options)
        self.assertEqual(options, {"TABLETS_ROUTING_V1": ""})

        self.assertFalse(ProtocolFeatures.parse_from_supported({}).tablets_routing_v1)
