import logging
import os
from typing import List, Optional

# flake8: noqa
import comet_ml

import pytorch_lightning as pl
from hydra.utils import call, instantiate
from omegaconf import DictConfig

from src.utils import template_utils

log = logging.getLogger(__name__)


def train(config: DictConfig) -> None:

    print(config.model)

    if "seed" in config:
        pl.seed_everything(config.seed)

    # Initialize datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>.")
    datamodule = instantiate(config.datamodule)

    # Initialize model
    log.info(f"Instantiating model <{config.model._target_}>.")
    model = instantiate(config.model, optimizer=config.optimizer, _recursive_=False)

    # Init callbacks (e.g. checkpoints, early stopping)
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>.")
                callbacks.append(instantiate(cb_conf))

    # Init logggers (e.g. comet-ml)
    loggers = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>.")
                loggers.append(instantiate(lg_conf))

    # Init trainer
    trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    # Send config params to loggers
    log.info("Logging hyperparameters.")

    template_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    if "test" not in config:

        # Train model
        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)

        # Evaluate model on test set
        log.info("Starting test set evaluation.")
        if config.trainer.get("fast_dev_run"):
            trainer.test(ckpt_path=None)
        else:
            trainer.test()

        # Print path to best checkpoint
        log.info(
            f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}"
        )

    else:
        # Test mode: model loaded from chekpoint
        log.info(
            f"Starting test set evaluation on checkpoint {config.model.checkpoint_path}"
        )
        trainer.test(model=model, datamodule=datamodule)
