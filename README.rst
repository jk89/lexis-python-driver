lexis-python-driver
===================

.. image:: https://raw.githubusercontent.com/jk89/lexisdb/develop/lexis-icon.svg
   :width: 160
   :height: 196
   :align: left
   :alt: Lexis

The Python driver for `Lexis <https://github.com/jk89/lexisdb>`_, a fork of
ScyllaDB that replaces the Murmur3 partitioner with byte-ordered partitioning,
built on tablets. In Lexis the token of a partition is an order-preserving
encoding of its partition key.

With a Lexis cluster, the driver computes these tokens on the client, so each
query is sent to the node and shard that own its partition, and it follows
Lexis tablet routing. ``token()`` values are returned as ``bytes``. With
Cassandra and ScyllaDB clusters the driver behaves as upstream.

Installation
------------

::

    pip install git+https://github.com/jk89/lexis-python-driver

The package is ``lexis-driver``; it is imported as ``cassandra``, like the
upstream driver, so uninstall ``scylla-driver`` or ``cassandra-driver`` first.

Base and licence
----------------

Forked from the ScyllaDB Python driver at tag ``3.26.9-scylla``. Licensed
under the Apache License 2.0 (``LICENSE``).

The upstream README is kept unchanged in ``README_SCYLLA.rst``.