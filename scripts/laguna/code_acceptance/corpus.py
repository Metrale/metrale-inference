# SPDX-License-Identifier: MIT OR Apache-2.0

"""2026-10-07: Authored semantic fixtures; candidate source is data, never host-executed."""


def case(args, expected=None, error=None):
    return {"args": args, "expected": expected, "error": error}


MERGE = """def solve(intervals):
    if any(len(x) != 2 or type(x[0]) is not int or type(x[1]) is not int or x[0] >= x[1] for x in intervals):
        raise ValueError('invalid interval')
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result
"""
RETRY = """def solve(status, header, attempt):
    if type(attempt) is not int or attempt < 0:
        raise ValueError('attempt')
    if status not in (429, 503):
        return None
    text = header.strip() if isinstance(header, str) else ''
    if text and all('0' <= c <= '9' for c in text):
        digits = text.lstrip('0') or '0'
        return 60 if len(digits) > 2 else min(60, int(digits))
    return 60 if attempt >= 6 else min(60, 2 ** attempt)
"""
TOPO = """def solve(graph):
    import heapq
    if any(dep not in graph for deps in graph.values() for dep in deps):
        raise ValueError('unknown')
    remaining = {name: set(deps) for name, deps in graph.items()}
    ready = [name for name, deps in remaining.items() if not deps]
    heapq.heapify(ready)
    result = []
    while ready:
        name = heapq.heappop(ready)
        result.append(name)
        for node, deps in remaining.items():
            if name in deps:
                deps.remove(name)
                if not deps:
                    heapq.heappush(ready, node)
    if len(result) != len(graph):
        raise ValueError('cycle')
    return result
"""
INGEST = """def solve(existing, incoming):
    import copy
    output = []
    seen = {}
    for event in existing + incoming:
        key = event['id']
        if key in seen:
            if event != seen[key]:
                raise ValueError('conflict')
        else:
            seen[key] = event
            output.append(copy.deepcopy(event))
    return output
"""
E1 = {"id": "a", "seq": 2, "payload": "one"}
E2 = {"id": "b", "seq": 1, "payload": "two"}
E3 = {"id": "c", "seq": 2, "payload": "three"}

CORPUS = [
    {
        "id": "merge_intervals",
        "role": "generation",
        "requirements": "Write Python 3 standard-library code defining solve(intervals). Input is a list of two-element integer lists representing half-open [start,end) intervals. Return a new list sorted by start, merging overlapping OR touching intervals. Empty input returns []. Duplicates collapse. Reject any start>=end or non-int endpoint (including bool) with ValueError. Do not mutate input. Output code only.",
        "tests": [
            case([[]], []),
            case([[[4, 7], [1, 3], [3, 5]]], [[1, 7]]),
            case([[[0, 10], [2, 3], [0, 10]]], [[0, 10]]),
            case([[[-5, -2], [0, 1]]], [[-5, -2], [0, 1]]),
            case([[[3, 3]]], error="ValueError"),
            case([[[5, 2]]], error="ValueError"),
            case([[[False, 2]]], error="ValueError"),
        ],
        "good": MERGE,
        "bad": {
            "touching_not_merged": MERGE.replace(
                "lo <= result[-1][1]", "lo < result[-1][1]"
            ),
            "sorts_input_in_place": MERGE.replace(
                "    result = []", "    intervals.sort()\n    result = []"
            ),
        },
    },
    {
        "id": "retry_delay_edit",
        "role": "edit",
        "requirements": "Repair the provided Python solve(status, header, attempt). attempt must be a nonnegative int (not bool), otherwise raise ValueError even for a non-retry status. Only statuses 429 and 503 retry; all others return None. For retry statuses, a string header containing only ASCII digits after stripping whitespace is an integer delay clamped to 60 seconds, including zero and arbitrarily many leading zeros. Missing/invalid headers (negative, decimal, Unicode digits, or nonstring) use min(60,2**attempt); huge attempt must complete without computing a huge power. Do not mutate arguments. Return the complete corrected code only.",
        "starter": "def solve(status, header, attempt):\n    return int(header) if header else 2 ** attempt\n",
        "tests": [
            case([200, "10", 0], None),
            case([429, " 0 ", 4], 0),
            case([503, "999", 0], 60),
            case([429, "0" * 5000 + "7", 2], 7),
            case([429, "-1", 2], 4),
            case([503, "1.5", 3], 8),
            case([429, "٢", 2], 4),
            case([503, None, 1000000000], 60),
            case([429, 12, 0], 1),
            case([200, None, -1], error="ValueError"),
            case([429, "", True], error="ValueError"),
        ],
        "good": RETRY,
        "bad": {
            "unicode_digits_accepted": RETRY.replace(
                "all('0' <= c <= '9' for c in text)", "text.isdigit()"
            ),
            "zero_replaced_with_backoff": RETRY.replace(
                "return 60 if len(digits) > 2 else min(60, int(digits))",
                "return (60 if len(digits) > 2 else min(60, int(digits))) or min(60, 2 ** attempt)",
            ),
        },
    },
    {
        "id": "stable_dependency_order",
        "role": "generation",
        "requirements": "Write Python solve(graph) for a dict mapping string node names to lists of dependencies. Return a topological order, choosing the lexicographically smallest currently ready node at every step. Repeated dependency names count once. Empty graph returns []. Raise ValueError for unknown dependency nodes, a self-cycle or any directed cycle (including a disconnected cycle). Do not mutate graph or its lists. Output code only.",
        "tests": [
            case([{}], []),
            case([{"b": [], "a": [], "c": ["a"]}], ["a", "b", "c"]),
            case([{"c": [], "a": ["b"], "b": []}], ["b", "a", "c"]),
            case([{"b": ["a", "a"], "a": []}], ["a", "b"]),
            case([{"a": ["missing"]}], error="ValueError"),
            case([{"a": ["a"]}], error="ValueError"),
            case([{"ok": [], "a": ["b"], "b": ["a"]}], error="ValueError"),
        ],
        "good": TOPO,
        "bad": {
            "wrong_ready_order": TOPO.replace(
                "name = heapq.heappop(ready)",
                "name = max(ready)\n        ready.remove(name)",
            ),
            "cycle_silently_truncated": TOPO.replace(
                "raise ValueError('cycle')", "return result"
            ),
        },
    },
    {
        "id": "idempotent_ingest_edit",
        "role": "edit",
        "requirements": "Repair Python solve(existing,incoming). Each argument is a list of events with exactly id(str),seq(int),payload(str), already valid. Return a new list in first-seen arrival order across existing then incoming. Identical events with the same id are retries and must appear once (including duplicates already in existing). If the same id has ANY different field, raise ValueError. Different IDs with equal seq remain separate; do not sort by seq. Do not mutate either input. Return complete corrected code only.",
        "starter": 'def solve(existing, incoming):\n    by_sequence = {event["seq"]: event for event in existing + incoming}\n    return [by_sequence[key] for key in sorted(by_sequence)]\n',
        "tests": [
            case([[], []], []),
            case([[E1], [E1, E2]], [E1, E2]),
            case([[E1, E1], [E3, E2, E3]], [E1, E3, E2]),
            case(
                [[E1], [{"id": "a", "seq": 2, "payload": "different"}]],
                error="ValueError",
            ),
            case(
                [[], [E1, {"id": "a", "seq": 99, "payload": "one"}]], error="ValueError"
            ),
            case([[E2, E1], []], [E2, E1]),
        ],
        "good": INGEST,
        "bad": {
            "conflicts_silently_ignored": INGEST.replace(
                "raise ValueError('conflict')", "pass"
            ),
            "sorts_by_sequence": INGEST.replace(
                "return output", "return sorted(output, key=lambda e: e['seq'])"
            ),
        },
    },
]
