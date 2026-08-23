#!/usr/bin/env python3
"""Generate deterministic exact-language fixtures for the controlled study."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
from pathlib import Path


def targets(bits: tuple[int, ...]) -> dict[str, int]:
    x = [bool(value) for value in bits]
    return {
        "unary": int(x[0]),
        "horn_simple": int((not x[0]) or x[1]),
        # Negative sets {}, {x0}, {x0,x1}: a strict star-nested chain.
        "horn_chain": int(x[3] and ((not x[0]) or x[4]) and ((not x[0]) or (not x[1]) or x[5])),
        # Exact polarity dual of horn_chain.
        "antihorn_chain": int((not x[3]) and (x[0] or (not x[4])) and (x[0] or x[1] or (not x[5]))),
        "square_form_i": int((x[0] or x[1]) and ((not x[2]) or x[3])),
        "square_form_ii": int(
            (x[0] or x[1]) and (x[1] or (not x[2])) and ((not x[2]) or x[3])
        ),
        "square_form_iii": int(
            (x[0] or x[1])
            and (x[1] or (not x[2]))
            and (x[0] or x[3])
            and ((not x[2]) or x[3])
        ),
        "affine_xor3": int(x[0] ^ x[1] ^ x[2]),
    }


def flip(target: str, noise_percent: int, row: int, repetition: int) -> bool:
    if noise_percent == 0:
        return False
    digest = hashlib.sha256(
        f"smart-mdt-controlled-v1:{target}:{noise_percent}:{row}:{repetition}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % 100 < noise_percent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    assignments = list(product([0, 1], repeat=6))
    names = list(targets(assignments[0]))
    for name in names:
        for noise in (0, 5, 10):
            lines: list[str] = []
            for repetition in range(args.repetitions):
                for row, bits in enumerate(assignments):
                    label = targets(bits)[name]
                    if flip(name, noise, row, repetition):
                        label = 1 - label
                    lines.append(" ".join(map(str, (label, *bits))))
            path = args.output / f"{name}__noise_{noise:02d}.dl8"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = args.output / "TARGETS.md"
    manifest.write_text(
        """# Controlled theorem-valid targets

- `unary`: `x0`
- `horn_simple`: `not x0 OR x1`
- `horn_chain`: `x3 AND (not x0 OR x4) AND (not x0 OR not x1 OR x5)`
- `antihorn_chain`: exact polarity dual of `horn_chain`
- `square_form_i`: `(x0 OR x1) AND (not x2 OR x3)`
- `square_form_ii`: `(x0 OR x1) AND (x1 OR not x2) AND (not x2 OR x3)`
- `square_form_iii`: `(x0 OR x1) AND (x1 OR not x2) AND (x0 OR x3) AND (not x2 OR x3)`
- `affine_xor3`: `x0 XOR x1 XOR x2`

Each target is emitted at 0%, 5%, and 10% deterministic label noise. The
complete six-variable Boolean domain is repeated eight times. Labels are
generated before any train/test split; no test label participates in candidate
construction.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
