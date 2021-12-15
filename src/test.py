from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)


def _locate(path: str) -> Union[type, Callable[..., Any]]:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".") if part]
    for n in reversed(range(1, len(parts) + 1)):
        mod = ".".join(parts[:n])
        try:
            obj = import_module(mod)
        except Exception as exc_import:
            if n == 1:
                raise ImportError(f"Error loading module '{path}'") from exc_import
            continue
        break
    for m in range(n, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    import_module(mod)
                except ModuleNotFoundError:
                    pass
                except Exception as exc_import:
                    raise ImportError(f"Error loading '{path}': '{repr(exc_import)}'") from exc_import
            raise ImportError(f"Encountered AttributeError while loading '{path}': {exc_attr}") from exc_attr
    if isinstance(obj, type):
        obj_type: type = obj
        return obj_type
    elif callable(obj):
        obj_callable: Callable[..., Any] = obj
        return obj_callable
    else:
        # reject if not callable & not a type
        raise ValueError(f"Invalid type ({type(obj)}) found for {path}")


def test(config: DictConfig) -> Optional[float]:
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
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model_cls = _locate(config.model._target_)
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
    trainer.test(model=model, datamodule=datamodule)

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
