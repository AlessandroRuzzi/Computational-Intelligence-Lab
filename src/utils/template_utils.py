"""
    File adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/main/src/utils/template_utils.py
    Helps setting up configurations (e.g. diable warnings, setting pytorch lightning in debug mode, logging, ...)
"""

import logging
import warnings
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.syntax import Syntax
from rich.tree import Tree

log = logging.getLogger(__name__)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file.
        - disabling warnings
        - easier access to debug mode
        - forcing debug friendly configuration
        - forcing multi-gpu friendly configuration
    Args:
        config (DictConfig): [description]
    """

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.disable_warnings=True>
    if config.get("disable_warnings"):
        log.info("Disabling python warnings! <config.disable_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        # if config.datamodule.get("num_workers"):
        #    config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    if config.trainer.get("accelerator") in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info("Forcing ddp friendly configuration! <config.trainer.accelerator=ddp>")
        # ddp doesn't like num_workers>0 or pin_memory=True
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "optimizer",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    print(tree)


def empty(*args: str, **kwargs: int) -> None:
    pass


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
        - sizes of train, val, test dataset
        - number of trainable model parameters

    Args:
        config (DictConfig): [description]
        model (pl.LightningModule): [description]
        datamodule (pl.LightningDataModule): [description]
        trainer (pl.Trainer): [description]
        callbacks (List[pl.Callback]): [description]
        logger (List[pl.loggers.LightningLoggerBase]): [description]
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "optimizer" in config:
        hparams["optimizer"] = config["optimizer"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save sizes of each dataset
    # (requires calling `datamodule.setup()` first to initialize datasets)
    # datamodule.setup()
    # if hasattr(datamodule, "data_train") and datamodule.data_train:
    #     hparams["datamodule/train_size"] = len(datamodule.data_train)
    # if hasattr(datamodule, "data_val") and datamodule.data_val:
    #     hparams["datamodule/val_size"] = len(datamodule.data_val)
    # if hasattr(datamodule, "data_test") and datamodule.data_test:
    #     hparams["datamodule/test_size"] = len(datamodule.data_test)

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = empty


def imshow(
    image: Any,
    normalize: bool = False,
) -> None:
    """Imshow for Tensor."""
    fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image, cmap="gray")
    ax.set_xticklabels(list(""))
    ax.set_yticklabels(list(""))
    plt.show()


def log_image(experiment: Any, image: torch.Tensor, title: str) -> None:
    image = torchvision.utils.make_grid(image)
    image = image.permute((1, 2, 0))
    # print(image.size())
    experiment.log_image(image.cpu(), title)
