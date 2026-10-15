-- SPDX-License-Identifier: MIT OR Apache-2.0

-- 2026-10-15: made once with `sqlite3 reader.sqlite < reader.sql`.
-- Exercises the file-format reader: 512-byte pages, every integer width,
-- floats, NULLs, a blob, a record shorter than its table (a column added
-- later), a multi-level table b-tree, and a TEXT value spread over an
-- overflow chain.
PRAGMA page_size = 512;
CREATE TABLE vals (id INTEGER PRIMARY KEY, n INTEGER, x REAL, t TEXT, b BLOB);
INSERT INTO vals VALUES
    (1, 0, 0.5, 'zero', NULL),
    (2, 1, -2.25, '', x'00ff10'),
    (3, 127, 1e300, NULL, NULL),
    (4, -128, NULL, 'i8', NULL),
    (5, 32767, 3.0, 'i16', NULL),
    (6, -32768, NULL, NULL, NULL),
    (7, 8388607, NULL, 'i24', NULL),
    (8, -8388608, NULL, NULL, NULL),
    (9, 2147483647, NULL, 'i32', NULL),
    (10, -2147483648, NULL, NULL, NULL),
    (11, 140737488355327, NULL, 'i48', NULL),
    (12, -140737488355328, NULL, NULL, NULL),
    (13, 9223372036854775807, NULL, 'i64', NULL),
    (14, -9223372036854775808, NULL, NULL, NULL),
    (15, NULL, NULL, NULL, NULL);
ALTER TABLE vals ADD COLUMN late INTEGER;
INSERT INTO vals VALUES (16, 2, NULL, 'late', NULL, 42);
CREATE TABLE many (k INTEGER NOT NULL, sq INTEGER NOT NULL);
WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM c WHERE i < 4000)
INSERT INTO many SELECT i, i * i FROM c;
CREATE TABLE big (id INTEGER PRIMARY KEY, body TEXT NOT NULL);
INSERT INTO big VALUES (1, 'short');
INSERT INTO big SELECT 2, group_concat(i, ',') FROM (
    WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM c WHERE i < 1500) SELECT i FROM c);
INSERT INTO big VALUES (3, 'after');
CREATE TABLE "quoted name" ("start" INTEGER, [end] INTEGER, `v` TEXT DEFAULT 'd');
INSERT INTO "quoted name" VALUES (5, 6, 'q');
