-- SPDX-License-Identifier: MIT OR Apache-2.0

-- 2026-10-15: made once with `sqlite3 wal.sqlite < wal.sql`. A database left in
-- WAL mode (header bytes 18/19 = 2); the reader must refuse it.
PRAGMA journal_mode = WAL;
CREATE TABLE t (v INTEGER);
INSERT INTO t VALUES (1);
