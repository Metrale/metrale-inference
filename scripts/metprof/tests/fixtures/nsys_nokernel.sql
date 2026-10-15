-- SPDX-License-Identifier: MIT OR Apache-2.0

-- 2026-10-15: made once with `sqlite3 nsys_nokernel.sqlite < nsys_nokernel.sql`.
-- An export with no kernel table.
CREATE TABLE StringIds (id INTEGER NOT NULL PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO StringIds VALUES (1, 'unused');
