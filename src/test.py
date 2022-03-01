from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from sklearn.model_selection import KFold

from src.train import PrepareTmpFile
from src.utils import utils

log = utils.get_logger(__name__)


def test(config: DictConfig, datamodule: Optional[LightningDataModule] = None) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    if datamodule is None:
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model_cls = utils._locate(config.model._target_)
    checkpoint_path: Path = Path(config.work_dir) / config.load_checkpoint
    model: LightningModule = model_cls.load_from_checkpoint(checkpoint_path)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Evaluate model on test set, using the best model achieved during training
    log.info("Starting testing!")
    result: List[Dict[str, float]] = trainer.test(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )
    return result


def test_cv(config: OmegaConf, df: pd.DataFrame):
    # Filter run
    log.debug("Filtering")
    log.debug(f"Length: {len(df)}")
    for name, d in [("model", config.model), ("dataset", config.datamodule), ("trainer", config.trainer)]:
        for k, v in d.items():
            if len(df) == 1:
                break
            df = df[df[f"{name}_{k}"] == v]
            log.debug(f"{name}_{k}={v}")
            log.debug(f"Length: {len(df)}")
    index = df.index
    assert len(index) == 1
    run_name = index[0]
    log.info(f"Run name: {run_name}")
    checkpoint_paths = df.filter(regex="^best_checkpoint")

    result_dict = defaultdict(list)

    # Load csv
    df = pd.read_csv(config.datamodule.csv_path)
    kf = KFold(n_splits=config["folds"], shuffle=True, random_state=config.seed)
    datamodule_params = dict(config.datamodule)
    datamodule_cls = utils._locate(datamodule_params.pop("_target_"))
    datamodule_params.pop("csv_path")  # remove csv_path from params
    for i, (checkpoint_path, (train_idx, test_idx)) in enumerate(
        zip(checkpoint_paths.values[0], kf.split(df)), start=1
    ):
        log.info(f"Start {i}th fold out of {kf.n_splits} folds")
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        valid_df, test_df = np.array_split(test_df, 2)

        log.info(checkpoint_path)
        config.load_checkpoint = checkpoint_path

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        with PrepareTmpFile(train_df, valid_df, test_df) as (ft, fv, fe):
            datamodule: LightningDataModule = datamodule_cls(ft.name, fv.name, fe.name, **datamodule_params)
            result: List[Dict[str, float]] = test(config, datamodule)
        print(result)
        assert len(result) == 1
        result = result[0]
        for k, v in result.items():
            result_dict[k].append(v)
    utils.log_cv_result(run_name, config, result_dict)
