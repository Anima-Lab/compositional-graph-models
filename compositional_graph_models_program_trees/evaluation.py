"""
Script to evaluate trained models.
"""

import logging
import pprint
from copy import copy

import hydra
import omegaconf
import pytorch_lightning as pl

from compositional_graph_models import data, models

_logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="evaluation")
def main(cfg: omegaconf.DictConfig):
    # Hydra handles logging
    for handler in pl._logger.handlers:
        pl._logger.removeHandler(handler)

    yaml_config = omegaconf.OmegaConf.to_yaml(cfg)
    _logger.info(f"Starting evaluation with config:\n{yaml_config}")

    _logger.info("Loading model and trainer...")
    model_class = hydra._internal.utils._locate(cfg.model_class)
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
    model = model_class.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(**cfg.trainer)

    _logger.info("Loading data...")
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    data_module = data.EquationTreeDataModule(
        train_path=None,
        val_path=None,
        test_path=data_path,
        function_vocab=model.hparams.function_vocab,
        token_vocab=model.hparams.token_vocab,
        **cfg.data,
    )

    _logger.info(f"Beginning evaluation on dataset: {cfg.data_path}")
    data_module.setup("test")
    data_loader = data_module.test_dataloader()
    trainer.test(model=model, test_dataloaders=data_loader)

    results = copy(model._results)
    del results["meta"]
    for k, v in results.items():
        results[k] = v.item()

    result_string = pprint.pformat(results, indent=2)
    _logger.info(f"Results:\n{result_string}")


if __name__ == "__main__":
    main()
