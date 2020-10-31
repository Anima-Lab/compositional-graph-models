import logging

import hydra
import omegaconf
import pytorch_lightning as pl

from compositional_graph_models import data, models

_logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="training")
def main(cfg: omegaconf.DictConfig):
    # Hydra handles logging
    for handler in pl._logger.handlers:
        pl._logger.removeHandler(handler)

    yaml_config = omegaconf.OmegaConf.to_yaml(cfg)
    _logger.info(f"Starting a training run with config:\n{yaml_config}")

    _logger.info("Loading training data...")
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.data.train_path = cfg.data.train_path and hydra.utils.to_absolute_path(
        cfg.data.train_path
    )
    cfg.data.val_path = cfg.data.val_path and hydra.utils.to_absolute_path(
        cfg.data.val_path
    )
    cfg.data.test_path = None
    data_module = data.EquationTreeDataModule(**cfg.data)
    data_module.setup("fit")

    _logger.info("Building model and trainer...")
    model = hydra.utils.instantiate(
        cfg.model,
        function_vocab=data_module.function_vocab,
        token_vocab=data_module.token_vocab,
    )
    trainer = pl.Trainer(**cfg.trainer)

    _logger.info("Beginning training")
    trainer.fit(model, datamodule=data_module)

    _logger.info("Done!")


if __name__ == "__main__":
    main()
