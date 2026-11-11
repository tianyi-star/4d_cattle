# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Index of walking sequences under ``data_root``.

The object follows the sequence side of the PyTorch dataset protocol
(``__len__`` and ``__getitem__``) without importing torch. Pixel decoding is
left to the experiment so the reader can be OpenCV, torchvision, or a custom
cropper around an ultralytics detection.
"""

from __future__ import annotations

from pathlib import Path

from cattle4d.data.multiview import load_sequence, validate_sequence
from cattle4d.data.schema import Sequence


class GaitSequenceDataset:
    """Sequences listed in ``<data_root>/sequences``.

    Parameters
    ----------
    data_root :
        Directory that contains ``sequences/`` and, optionally, ``splits/``.
    split :
        When set, keep only ids listed in ``splits/<split>.txt``. Lines
        starting with ``#`` are ignored. When omitted, every valid sequence
        directory is returned.
    """

    def __init__(self, data_root: Path, split: str | None = None) -> None:
        self.data_root = Path(data_root)
        self.split = split
        sequence_root = self.data_root / "sequences"
        if not sequence_root.is_dir():
            raise FileNotFoundError(
                f"No sequences directory at {sequence_root}. "
                "Create the layout documented in README.md and set data_root "
                "in configs/path.yaml."
            )
        discovered = sorted(
            path.name for path in sequence_root.iterdir() if path.is_dir()
        )
        if split is not None:
            allow = _read_split(self.data_root / "splits" / f"{split}.txt")
            discovered = [name for name in discovered if name in allow]
        if not discovered:
            raise FileNotFoundError(
                f"No sequences found under {sequence_root} for split {split!r}."
            )
        broken: list[str] = []
        for name in discovered:
            issues = validate_sequence(sequence_root / name)
            if issues:
                broken.append(name + ": " + "; ".join(issues))
        if broken:
            raise ValueError(
                "Sequence layout check failed.\n" + "\n".join(broken)
            )
        self.sequence_root = sequence_root
        self.ids = discovered

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        return self.load(index).to_dict()

    def load(self, index: int) -> Sequence:
        if index < 0 or index >= len(self.ids):
            raise IndexError(f"Sequence index {index} is outside 0..{len(self.ids) - 1}.")
        return load_sequence(self.sequence_root / self.ids[index])


def _read_split(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            names.add(item)
    if not names:
        raise ValueError(f"Split file {path} has no sequence ids.")
    return names
