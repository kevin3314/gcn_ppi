import logging
import random
import warnings
from typing import Any, Callable, Dict, List, Sequence, Union

import mlflow
import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # if <config.name=...>
    if config.get("name"):
        log.info("Running in experiment mode! Name: {}".format(config.name))

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


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


def get_run_name(conf: OmegaConf) -> str:
    """Returns run name based on the config."""

    model_conf = conf.model
    result = model_conf._target_.split(".")[-1]
    # Add random hash
    return result + str(random.getrandbits(32))


def get_experiment_name(config: OmegaConf) -> str:
    """Returns experiment name based on the config."""
    result = config.experiment_name
    if config.do_cross_validation:
        result = result + "_cv"
    else:
        result = result + "_wo_cv"
    return result


@rank_zero_only
def log_result(run_name: str, config: OmegaConf, res_dict: Dict[str, Any], best_paths: List[str]) -> None:
    experiment_name = get_experiment_name(config)
    # Log/Print results
    mlflow.set_tracking_uri(f"file://{HydraConfig.get().runtime.cwd}/mlruns")
    mlflow.set_experiment(experiment_name)
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"run name: {run_name}")
    logger.info("-" * 60)
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        for name, d in [("model", config.model), ("dataset", config.datamodule), ("trainer", config.trainer)]:
            for k, v in d.items():
                mlflow.log_param(f"{name}_{k}", v)

        for i, checkpoints in enumerate(best_paths):
            mlflow.log_param(f"best_checkpoint_{i}fold", checkpoints)

        # Log metrics
        for metric, res in res_dict.items():
            logger.info(f"All:     {metric} = {res}")
            logger.info(f"Average: {metric} = {np.mean(np.array(res))}")
            logger.info(f"Std:     {metric} = {np.std(np.array(res))}")
            mlflow.log_metric(f"{metric}_mean", np.mean(np.array(res)))
            mlflow.log_metric(f"{metric}_std", np.std(np.array(res)))


def get_cv_test_experiment_name(config: OmegaConf) -> str:
    """Returns experiment name based on the config."""
    result = config.experiment_name
    return result + "_cv_test"


@rank_zero_only
def log_cv_result(run_name: str, config: OmegaConf, res_dict: Dict[str, Any]):
    experiment_name = get_cv_test_experiment_name(config)
    # Log/Print results
    mlflow.set_tracking_uri(f"file://{HydraConfig.get().runtime.cwd}/mlruns")
    mlflow.set_experiment(experiment_name)
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"run name: {run_name}")
    logger.info("-" * 60)
    # Log metrics
    with mlflow.start_run(run_name=run_name):
        for metric, res in res_dict.items():
            logger.info(f"All:     {metric} = {res}")
            logger.info(f"Average: {metric} = {np.mean(np.array(res))}")
            logger.info(f"Std:     {metric} = {np.std(np.array(res))}")
            mlflow.log_metric(f"{metric}_mean", np.mean(np.array(res)))
            mlflow.log_metric(f"{metric}_std", np.std(np.array(res)))
