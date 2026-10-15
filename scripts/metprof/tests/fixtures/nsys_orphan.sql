-- SPDX-License-Identifier: MIT OR Apache-2.0

-- 2026-10-15: made once with `sqlite3 nsys_orphan.sqlite < nsys_orphan.sql`.
-- A kernel whose shortName id (99) is absent from StringIds.
CREATE TABLE StringIds (id INTEGER NOT NULL PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO StringIds VALUES (1, 'gemm_a');
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER NOT NULL, end INTEGER NOT NULL,
    deviceId INTEGER NOT NULL, streamId INTEGER NOT NULL, correlationId INTEGER,
    shortName INTEGER NOT NULL, gridX INTEGER NOT NULL, gridY INTEGER NOT NULL, gridZ INTEGER NOT NULL);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 10, 0, 1, 1, 1, 1, 1, 1), (20, 30, 0, 1, 2, 99, 1, 1, 1);
