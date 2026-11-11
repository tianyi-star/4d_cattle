# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Training entry point.

``python -m cattle4d.engine.train --dry-run`` builds the pipeline and prints
its contract. ``fit`` is where the loss and the optimizer step go. It refuses
to update parameters, and it writes no checkpoint.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from cattle4d.config import ExperimentConfig, load_experiment
from cattle4d.data.dataset import GaitSequenceDataset
from cattle4d.pipeline import Cattle4DPipeline, build_pipeline

logger = logging.getLogger("cattle4d.train")


def fit(pipeline: Cattle4DPipeline, dataset: GaitSequenceDataset, config: dict[str, Any]) -> None:
    """Optimization entry point.

    Parameters
    ----------
    pipeline :
        Stages from ``build_pipeline``. Their ``forward`` methods are still
        unimplemented, so this function does not call them.
    dataset :
        Indexed sequences. Collation and pixel loading belong with the loss.
    config :
        The ``optim`` block must set ``lr`` and ``epochs`` before a real run.
        Those fields ship as null because no schedule has been selected.

    The function checks that a schedule was chosen and then stops. Add the
    objective (photometric replay, forecast, and recognition) and the
    optimizer step in this body. Do not drop in an external checkpoint as a
    substitute for that experiment.
    """

    if len(dataset) == 0:
        raise ValueError("The training split is empty.")
    optim = config.get("optim", {})
    if optim.get("lr") is None or optim.get("epochs") is None:
        raise ValueError(
            "Set optim.lr and optim.epochs in the experiment config before training. "
            "This repository does not ship a learning-rate schedule."
        )
    raise NotImplementedError(
        "cattle4d.engine.train.fit has no loss and no optimizer step. "
        "Implement the walking-replay, prediction, and recognition objectives here. "
        "No checkpoint will be written until that step exists."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    experiment = load_experiment(args.config, args.paths)
    pipeline = build_pipeline(experiment.config)
    logger.info("Cattle4DGS training scaffold")
    logger.info(pipeline.describe())
    logger.info("trainable parameter count: %d", pipeline.trainable_parameter_count())
    if args.dry_run:
        logger.info("Dry run finished. fit() was not called.")
        return 0
    dataset = _open_split(experiment, args.split)
    fit(pipeline, dataset, experiment.config)
    return 0


def _open_split(experiment: ExperimentConfig, split: str) -> GaitSequenceDataset:
    root = experiment.data_root
    if not root.is_dir():
        raise FileNotFoundError(
            f"data_root is {root}. Point configs/path.yaml at a local sequence "
            "tree before training. --dry-run checks the pipeline without data."
        )
    return GaitSequenceDataset(root, split=split)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cattle4d.engine.train",
        description="Train the Cattle4DGS gait pipeline. The update step is unimplemented.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML.")
    parser.add_argument("--paths", type=Path, required=True, help="Local path YAML.")
    parser.add_argument(
        "--split",
        default="train",
        help="Split name. Reads <data_root>/splits/<split>.txt when training for real.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the module graph and exit before any data load or update.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
