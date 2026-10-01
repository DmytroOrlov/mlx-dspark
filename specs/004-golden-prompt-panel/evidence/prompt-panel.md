# Golden Prompt Panel Results

Each prompt is shown individually. No aggregate throughput is used as a headline.

The first two attempts failed before usable measurements; their failure artifacts remain preserved in the JSON observation ledger. Corrected attempt 3 supplies the results below. Frozen physical runner SHA-256: `f80ba3c06517f5faa007af3086fe651aeba4aa5a5ea58cb54da0f38df3d6b984`.

Screen: 17 prompts; 12 naturally generated at least 410 tokens; 9 of those reached 40 tok/s in the screen; 9 new prompts certified.

Historical LIS/P02 qualifies only if its current screen and fresh repeat both independently meet the 410-token and 40-tok/s thresholds.

The historical LIS/P02 prompt does not certify under this current >=410-token protocol.
Certified new prompts: P05, P06, P07, P08, P09, P10, P14, P15, P17.
Screen >=40 tok/s and >=410 tokens: P05, P06, P07, P08, P09, P10, P14, P15, P17.
If several prompts appear in these lists, fast speculative decoding spans several coding tasks; if the list is concentrated, that is prompt sensitivity.

## CERTIFIED GOLDEN

| Prompt | Short label | Generated tokens | Spec tok/s run 1 | Spec tok/s run 2 | Minimum certified tok/s | Acceptance | Target forwards | Generated/forward | Widths | Peak GiB | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| P05: Write a Python function that merges two sorted lists into a single sorted list. Include tests and a short complexity analysis. | Merge sorted lists | 513 | 52.798 | 55.533 | 52.798 | 0.880 | 75 | 6.840 | {'2': 4, '7': 70} | 15.918 | CERTIFIED GOLDEN |
| P06: Write a Python function that finds the maximum subarray sum using Kadane's algorithm. Explain why the algorithm runs in O(n) time. | Maximum subarray | 512 | 40.664 | 43.352 | 40.664 | 0.645 | 96 | 5.333 | {'2': 4, '7': 91} | 15.923 | CERTIFIED GOLDEN |
| P07: Write a Python function that returns the first non-repeating character in a string. Include type hints and unit tests. | First unique character | 514 | 45.211 | 45.778 | 45.211 | 0.693 | 91 | 5.648 | {'2': 4, '7': 86} | 15.923 | CERTIFIED GOLDEN |
| P08: Write a production-quality Python implementation of a stack with push, pop, peek, and is_empty methods. Include type hints and tests. | Stack | 517 | 48.144 | 47.854 | 47.854 | 0.739 | 87 | 5.943 | {'2': 4, '7': 82} | 15.919 | CERTIFIED GOLDEN |
| P09: Write a Python function that determines whether two strings are anagrams. Include tests and explain the time complexity. | Anagrams | 517 | 47.000 | 46.760 | 46.760 | 0.718 | 89 | 5.809 | {'2': 4, '7': 84} | 15.922 | CERTIFIED GOLDEN |
| P10: Write a Python function that finds the kth largest element in a list. Use an efficient algorithm and explain its expected time complexity. | Kth largest | 514 | 44.527 | 42.589 | 42.589 | 0.675 | 93 | 5.527 | {'2': 4, '7': 88} | 15.923 | CERTIFIED GOLDEN |
| P14: Write a Python function that computes the edit distance between two strings using dynamic programming. Include tests and explain the time and space complexity. | Edit distance | 516 | 51.113 | 51.594 | 51.113 | 0.844 | 78 | 6.615 | {'2': 4, '7': 73} | 15.918 | CERTIFIED GOLDEN |
| P15: Write a Python function that groups a list of strings into groups of anagrams. Include type hints, tests, and complexity analysis. | Group anagrams | 515 | 42.524 | 42.896 | 42.524 | 0.676 | 93 | 5.538 | {'2': 4, '7': 88} | 15.924 | CERTIFIED GOLDEN |
| P17: Write a Python function that performs topological sorting on a directed acyclic graph. Include cycle detection, type hints, and tests. | Topological sort | 518 | 44.017 | 45.148 | 44.017 | 0.720 | 89 | 5.820 | {'2': 4, '7': 84} | 15.919 | CERTIFIED GOLDEN |

## GOLDEN CANDIDATE awaiting repeat

| Prompt | Short label | Generated tokens | Spec tok/s run 1 | Spec tok/s run 2 | Minimum certified tok/s | Acceptance | Target forwards | Generated/forward | Widths | Peak GiB | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|

## SLOW but long enough

| Prompt | Short label | Generated tokens | Spec tok/s run 1 | Spec tok/s run 2 | Minimum certified tok/s | Acceptance | Target forwards | Generated/forward | Widths | Peak GiB | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| P01: Write a production-quality Python LRU cache with tests and type hints. | Python LRU cache | 519 | 39.295 | — | — | 0.639 | 98 | 5.296 | {'2': 4, '7': 93} | 15.920 | SLOW but long enough |
| P11: Write a production-quality Python implementation of a queue using two stacks. Include type hints, docstrings, and tests. | Queue using two stacks | 513 | 38.323 | — | — | 0.598 | 102 | 5.029 | {'2': 4, '7': 97} | 15.918 | SLOW but long enough |
| P16: Write a production-quality Python thread-safe singleton class. Include type hints, documentation, and tests. | Thread-safe singleton | 515 | 35.768 | — | — | 0.545 | 110 | 4.682 | {'2': 4, '7': 105} | 15.922 | SLOW but long enough |

## TOO SHORT

| Prompt | Short label | Generated tokens | Spec tok/s run 1 | Spec tok/s run 2 | Minimum certified tok/s | Acceptance | Target forwards | Generated/forward | Widths | Peak GiB | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| P02: Write a Python function that returns the length of the longest increasing subsequence. Include a short explanation of its time complexity. | Longest increasing subsequence | 370 | 44.003 | — | — | 0.715 | 65 | 5.692 | {'2': 4, '7': 60} | 15.904 | TOO SHORT |
| P03: Write a Python function that checks whether a string is a palindrome. Include a short explanation of its time and space complexity. | Palindrome | 209 | 46.249 | — | — | 0.720 | 38 | 5.500 | {'2': 4, '7': 33} | 15.883 | TOO SHORT |
| P04: Write a Python function that performs binary search on a sorted list. Include type hints and explain its time complexity. | Binary search | 226 | 48.817 | — | — | 0.764 | 39 | 5.795 | {'2': 4, '7': 34} | 15.883 | TOO SHORT |
| P12: Write a Python function that detects whether a linked list contains a cycle. Use O(1) extra space and explain the algorithm. | Linked-list cycle | 273 | 45.999 | — | — | 0.752 | 47 | 5.809 | {'2': 4, '7': 42} | 15.899 | TOO SHORT |
| P13: Write a Python function that returns the level-order traversal of a binary tree. Include type hints and a short complexity analysis. | Tree traversal | 375 | 45.443 | — | — | 0.713 | 66 | 5.682 | {'2': 4, '7': 61} | 15.903 | TOO SHORT |

## FAILED

| Prompt | Short label | Generated tokens | Spec tok/s run 1 | Spec tok/s run 2 | Minimum certified tok/s | Acceptance | Target forwards | Generated/forward | Widths | Peak GiB | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|

## P01 drift anchors

Raw P01 observations are retained and not used to normalize candidate rates.

| Role | Sequence | Generated | Spec tok/s | Input SHA-256 | Raw evidence |
|---|---:|---:|---:|---|---|
| screen | 1 | 519 | 39.295 | 8bda1dedde8e022452b7dafaa3677dca47dedc350258c0885c4d1e5fde155f59 | `specs/004-golden-prompt-panel/evidence/screen-attempt-03/01-P01-screen.json` |
| anchor-mid | 10 | 519 | 40.518 | 8bda1dedde8e022452b7dafaa3677dca47dedc350258c0885c4d1e5fde155f59 | `specs/004-golden-prompt-panel/evidence/screen-attempt-03/10-P01-anchor-mid.json` |
| anchor-end | 19 | 519 | 39.611 | 8bda1dedde8e022452b7dafaa3677dca47dedc350258c0885c4d1e5fde155f59 | `specs/004-golden-prompt-panel/evidence/screen-attempt-03/19-P01-anchor-end.json` |
