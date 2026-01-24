"""Power-of-2 encoding and constraint group definitions for Sudoku."""

from __future__ import annotations

# Map Sudoku digits 1-9 to powers of 2
ALLOWED_VALUES = tuple(2**i for i in range(9))  # (1, 2, 4, 8, 16, 32, 64, 128, 256)
TARGET_SUM = 511  # 2^9 - 1 = sum of all allowed values


def digit_to_power(d: int) -> int:
    """Map Sudoku digit 1-9 to 2^(d-1)."""
    if not 1 <= d <= 9:
        raise ValueError(f"Digit must be 1-9, got {d}")
    return 2 ** (d - 1)


def power_to_digit(p: int) -> int:
    """Map power-of-2 back to digit 1-9."""
    if p not in ALLOWED_VALUES:
        raise ValueError(f"Power must be in {ALLOWED_VALUES}, got {p}")
    return p.bit_length()


def get_constraint_groups() -> list[list[tuple[int, int]]]:
    """
    Return list of 27 constraint groups (9 rows, 9 columns, 9 boxes).
    Each group is a list of 9 (row, col) tuples.
    """
    groups = []

    # 9 row constraints
    for r in range(9):
        groups.append([(r, c) for c in range(9)])

    # 9 column constraints
    for c in range(9):
        groups.append([(r, c) for r in range(9)])

    # 9 box constraints (3x3 blocks)
    for box_r in range(3):
        for box_c in range(3):
            box = []
            for dr in range(3):
                for dc in range(3):
                    box.append((box_r * 3 + dr, box_c * 3 + dc))
            groups.append(box)

    return groups


# Precompute constraint groups (used throughout)
CONSTRAINT_GROUPS = get_constraint_groups()


def get_cell_groups(r: int, c: int) -> list[int]:
    """Return indices of the 3 constraint groups that contain cell (r, c)."""
    row_idx = r
    col_idx = 9 + c
    box_idx = 18 + (r // 3) * 3 + (c // 3)
    return [row_idx, col_idx, box_idx]
