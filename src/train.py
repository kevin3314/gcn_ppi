import tempfile
from collections import defaultdict
from typing import List, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from optuna import logging
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from sklearn.model_selection import KFold

from src.utils import utils

log = utils.get_logger(__name__)
log.setLevel(logging.INFO)


def train(config: DictConfig) -> Optional[float]:
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

    # Load csv
    df = pd.read_csv(config.datamodule.csv_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=config.seed)

    datamodule_params = dict(config.datamodule)
    datamodule_cls = utils._locate(datamodule_params.pop("_target_"))
    datamodule_params.pop("csv_path")  # remove csv_path from params

    res_dict = defaultdict(list)

    for i, (train, test) in enumerate(kf.split(df)):
        log.info(f"Start {i}th fold out of {kf.n_splits} folds")
        train_df = df.iloc[train]
        test_df = df.iloc[test]
        valid_df, test_df = np.array_split(test_df, 2)

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        ft = tempfile.NamedTemporaryFile()
        fv = tempfile.NamedTemporaryFile()
        fe = tempfile.NamedTemporaryFile()
        train_df.to_csv(ft.name, index=False)
        ft.seek(0)
        valid_df.to_csv(fv.name, index=False)
        fv.seek(0)
        test_df.to_csv(fe.name, index=False)
        fe.seek(0)
        datamodule: LightningDataModule = datamodule_cls(ft.name, fv.name, fe.name, **datamodule_params)

        # Init lightning model
        log.info(f"Instantiating model <{config.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(config.model)

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
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

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

        # Train the model
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        # Evaluate model on test set, using the best model achieved during training
        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            results = trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path)
            for metric in config.metrics:
                res_dict[metric].append(results[0][metric])

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

        # Print path to best checkpoint
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        optimized_metric = config.get("optimized_metric")
        if optimized_metric:
            return trainer.callback_metrics[optimized_metric]

        ft.close()
        fv.close()
        fe.close()

    # Print results
    log.info("-" * 60)
    for metric, res in res_dict.items():
        log.info(f"All:     {metric} = {res}")
        log.info(f"Average: {metric} = {np.mean(np.array(res))}")
