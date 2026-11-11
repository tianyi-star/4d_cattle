# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Load the experiment and path YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Parsed contents of ``configs/default.yaml`` and ``configs/path.yaml``."""

    config_path: Path
    paths_path: Path
    config: dict[str, Any]
    paths: dict[str, Any]

    @property
    def data_root(self) -> Path:
        return Path(self.paths["data_root"])

    @property
    def output_dir(self) -> Path:
        return Path(self.paths["output_dir"])

    @property
    def mmcows_root(self) -> Path | None:
        raw = self.paths.get("mmcows_root")
        if raw in (None, "", "/path/to/mmcows"):
            return None
        return Path(raw)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping. A missing file or a non-mapping document errors."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return document


def load_experiment(config_path: Path, paths_path: Path) -> ExperimentConfig:
    """Load both YAML files and check the keys the entry points rely on."""

    config = load_yaml(config_path)
    paths = load_yaml(paths_path)
    _require_keys(config, ("data", "model", "optim"), config_path)
    _require_keys(config["model"], ("coat_embed_dim", "gaussian", "recognizer"), config_path)
    _require_keys(config["model"]["gaussian"], ("sh_degree",), config_path)
    _require_keys(config["model"]["recognizer"], ("labels",), config_path)
    _require_keys(paths, ("data_root", "output_dir"), paths_path)
    labels = config["model"]["recognizer"]["labels"]
    if labels is None:
        config["model"]["recognizer"]["labels"] = []
    elif not isinstance(labels, list):
        raise ValueError("model.recognizer.labels must be a list of names.")
    return ExperimentConfig(
        config_path=Path(config_path),
        paths_path=Path(paths_path),
        config=config,
        paths=paths,
    )


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], origin: Path) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{origin} is missing keys: {', '.join(missing)}")
