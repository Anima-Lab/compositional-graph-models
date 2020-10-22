"""
Implements models for the equation verification task.

Models are implemented using DGL. See `data.py` for graph format.
"""

from typing import Dict, List

import dgl
import pytorch_lightning as pl
import torch

import data


class UnaryFunctionModule(torch.nn.Module):
    """
    An MLP block for single embedding vectors, producing vectors of the same shape.

    Parameters
    ----------
    d_model
        Dimensionality of the input, hidden layers, and output.
    num_layers
        How many dense layers to apply.
    """

    def __init__(self, d_model: int, num_layers: int):
        if num_layers <= 0:
            raise ValueError("A module must have at least one layer.")

        super().__init__()

        layers: List[torch.nn.Module] = []
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(d_model, d_model))
            layers.append(torch.nn.Tanh())

        self.layer_stack = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(inputs)


class BinaryFunctionModule(torch.nn.Module):
    """
    An MLP block for pairs of embedding vectors, producing single embedding vectors.

    Parameters
    ----------
    d_model
        Dimensionality of the input, hidden layers, and output.
    num_layers
        How many dense layers to apply.
    """

    def __init__(self, d_model: int, num_layers: int):
        if num_layers <= 0:
            raise ValueError("A module must have at least one layer.")

        super().__init__()

        layers: List[torch.nn.Module] = []
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(2 * d_model, 2 * d_model))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(2 * d_model, d_model))
        layers.append(torch.nn.Tanh())
        self.layer_stack = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(inputs)


# TODO: Generalize this into a BinaryTree base class and TreeRNN/TreeLSTM/... subclasses
class TreeRNN(pl.LightningModule):
    """
    An N-ary TreeRNN, using a different module for each function type.

    Tokens are indexed by number rather than name, so you must take care to use the
    *same* vocabulary each time you use a model. By default, vocabularies will be
    saved along with the model checkpoints, you just need to make sure to keep using
    the same ones.

    Parameters
    ----------
    d_model
        Size of the embedding vectors and hidden layers.
    num_module_layers
        Number of hidden layers for each module.
    learning_rate
        Learning rate to use for training.
    function_vocab
        Mapping from function names to indices.
    token_vocab
        Mapping from token names to indices.
    """

    def __init__(
        self,
        d_model: int,
        num_module_layers: int,
        learning_rate: float,
        function_vocab: Dict[str, int],
        token_vocab: Dict[str, int],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hparams.function_vocab_inverse = {v: k for k, v in function_vocab.items()}
        self.hparams.token_vocab_inverse = {v: k for k, v in token_vocab.items()}

        self.token_embedding = torch.nn.Embedding(
            num_embeddings=len(token_vocab),
            embedding_dim=d_model,
            padding_idx=token_vocab[data.NONLEAF_TOKEN],
        )
        self.unary_function_modules = torch.nn.ModuleDict(
            {
                f: UnaryFunctionModule(d_model, num_module_layers)
                for f in data.UNARY_FUNCTIONS
            }
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {
                f: BinaryFunctionModule(d_model, num_module_layers)
                for f in data.BINARY_FUNCTIONS
            }
        )
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))

        self.train_acc_metric = pl.metrics.Accuracy()
        self.val_acc_metric = pl.metrics.Accuracy()
        self.test_acc_metric = pl.metrics.Accuracy()

    def _nodes_function_type(self, nodes):
        # All nodes are the same type, so just check the first type
        function_idx = nodes.data[data.FUNCTION_FIELD][0].item()
        return self.hparams.function_vocab_inverse[function_idx]

    def _embed_nodes(self, nodes):
        """
        An "apply_node_func" for embedding leaf nodes. Does nothing to non-leaf nodes.
        """
        function_type = self._nodes_function_type(nodes)

        if function_type == data.LEAF_FUNCTION:
            embeddings = self.token_embedding(nodes.data[data.TOKEN_FIELD])
            return {"embed": embeddings}

        if function_type == data.EQUALITY_FUNCTION:
            return {"logits": nodes.data["logits"]}

        return {"embed": nodes.data["embed"]}

    def _reduce_func(self, nodes):
        """
        A "reduce_func" for receiving messages from children and computing activations.
        Does nothing to leaf nodes.
        """
        function_type = self._nodes_function_type(nodes)

        if function_type == data.LEAF_FUNCTION:
            return {}

        if function_type == data.EQUALITY_FUNCTION:
            left = nodes.mailbox["embed"][:, 0, :]
            right = nodes.mailbox["embed"][:, 1, :]
            logits = (left * right).sum(-1) + self.output_bias
            return {"logits": logits}

        if function_type in data.UNARY_FUNCTIONS:
            inputs = nodes.mailbox["embed"][:, 0, :]
            module = self.unary_function_modules[function_type]
            return {"embed": module(inputs)}

        if function_type in data.BINARY_FUNCTIONS:
            # Concatenate left and right messages along the embedding axis
            concatenated_inputs = nodes.mailbox["embed"].view(len(nodes), -1)
            module = self.binary_function_modules[function_type]
            return {"embed": module(concatenated_inputs)}

        assert False

    def forward(self, forest: dgl.DGLGraph) -> torch.Tensor:
        """
        Given a (possibly batched) forest, compute the logits that equality holds
        for each tree in the forest.
        """
        node_order = data.typed_topological_nodes_generator(forest)
        dgl.prop_nodes(
            forest,
            node_order,
            message_func=dgl.function.copy_src("embed", "embed"),
            reduce_func=self._reduce_func,
            apply_node_func=self._embed_nodes,
        )
        root_mask = (
            forest.ndata[data.FUNCTION_FIELD]
            == self.hparams.function_vocab[data.EQUALITY_FUNCTION]
        )
        logits = forest.ndata["logits"][root_mask]
        return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        forest, labels = batch
        logits = self(forest)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits, target=labels
        )
        self.train_acc_metric(logits, labels)

        self.log("loss/train", loss, on_step=True, on_epoch=False)
        self.log("accuracy/train", self.train_acc_metric, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        forest, labels = batch
        logits = self(forest)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits, target=labels
        )
        self.val_acc_metric(logits, labels)

        self.log("loss/val", loss, on_step=False, on_epoch=True)
        self.log("accuracy/val", self.val_acc_metric, on_step=False, on_epoch=True)

        # A placeholder metric used to compare results across multiple runs
        # with the tensorboard hparams dashboard. See:
        # https://pytorch-lightning.readthedocs.io/en/latest/logging.html#pytorch_lightning.loggers.tensorboard.TensorBoardLogger.params.default_hp_metric
        self.log("hp_metric", self.val_acc_metric, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx) -> None:
        forest, labels = batch
        logits = self(forest)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits, target=labels
        )
        self.test_acc_metric(logits, labels)

        self.log("loss/val", loss, on_step=False, on_epoch=True)
        self.log("accuracy/val", self.test_acc_metric, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
