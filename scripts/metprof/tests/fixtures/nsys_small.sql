-- SPDX-License-Identifier: MIT OR Apache-2.0

-- 2026-10-15: made once with `sqlite3 nsys_small.sqlite < nsys_small.sql`.
-- The tables and columns metprof reads, with the nsys names and nsys-style
-- DDL comments. Times are ns.
CREATE TABLE StringIds (
    -- Consolidation of repetitive string values.
    id                          INTEGER   NOT NULL   PRIMARY KEY,   -- ID reference value.
    value                       TEXT      NOT NULL                  -- String value.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start                       INTEGER   NOT NULL,                 -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                 -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                 -- Device ID.
    contextId                   INTEGER   NOT NULL,
    streamId                    INTEGER   NOT NULL,
    correlationId               INTEGER,
    demangledName               INTEGER   NOT NULL,                 -- REFERENCES StringIds(id)
    shortName                   INTEGER   NOT NULL,                 -- REFERENCES StringIds(id)
    gridX                       INTEGER   NOT NULL,
    gridY                       INTEGER   NOT NULL,
    gridZ                       INTEGER   NOT NULL,
    graphNodeId                 INTEGER                             -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
    start                       INTEGER   NOT NULL,
    end                         INTEGER   NOT NULL,
    deviceId                    INTEGER   NOT NULL,
    bytes                       INTEGER   NOT NULL,
    copyKind                    INTEGER   NOT NULL
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start                       INTEGER   NOT NULL,
    end                         INTEGER   NOT NULL,
    eventClass                  INTEGER   NOT NULL,
    correlationId               INTEGER,
    nameId                      INTEGER   NOT NULL                  -- REFERENCES StringIds(id)
);
INSERT INTO StringIds VALUES (1, 'gemm_b'), (2, 'gemm_a'), (3, 'nccl_x'), (4, 'cudaStreamSynchronize_v3020'),
    (5, 'void gemm_b<128>(float*)'), (6, 'void gemm_a<64>(float*)'), (7, 'ncclDevKernel_x(ncclWork)');
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
    (50000, 60000, 0, 1, 3, 77, 5, 1, 4, 2, 1, 9),
    (10000, 20000, 0, 1, 3, 1001, 6, 2, 1, 1, 1, NULL),
    (15000, 40000, 1, 2, 5, 1002, 7, 3, 1, 1, 1, NULL);
INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
    (0, 5000, 0, 64, 1),
    (70000, 71000, 0, 64, 2),
    (80000, 90000, 1, 64, 8);
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (61000, 69000, 1, 5, 4);
