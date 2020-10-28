"""
Implements models for the equation verification task.

Models are implemented using DGL. See `data.py` for graph format.
"""

import abc
from typing import Dict, List

import dgl
import pytorch_lightning as pl
import torch
import torch_scatter

from compositional_graph_models import data


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
        Dimensionality of the output. The input is expected to be twice this size.
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


class RecursiveNN(pl.LightningModule, abc.ABC):
    """
    A base class for unidirectional tree-structured models, which recursively compute
    the representation for each subtree from some function of the representation of the
    children of that subtree's root.

    This class makes the following assumptions about the model:
        - Leaves are always computed by an embedding lookup, and this may be done
          as the first step.
        - The root function is always the same.
        - Nodes pass fixed-size vectors along tree edges.
        - There is a fixed vocabulary for functions and tokens, known ahead of time.

    In exchange this class efficiently automates message passing for these models.

    NOTE: any superclass must call `self.save_hyperparameters()` before invoking
    this constructor or using any class methods.

    Parameters
    ----------
    d_model
        The dimension of the vectors passed along edges of the tree.
    learning_rate
        Learning rate to use for training.
    function_vocab
        Vocabulary mapping function names to indices.
    token_vocab
        Vocabulary mapping token names to indices.
    """

    def __init__(
        self,
        d_model: int,
        learning_rate: float,
        function_vocab: Dict[str, int],
        token_vocab: Dict[str, int],
    ):
        super().__init__()

        self.hparams.function_vocab_inverse = {v: k for k, v in function_vocab.items()}
        self.hparams.token_vocab_inverse = {v: k for k, v in token_vocab.items()}

        self.token_embedding = torch.nn.Embedding(
            num_embeddings=len(token_vocab),
            embedding_dim=d_model,
            padding_idx=token_vocab[data.NONLEAF_TOKEN],
        )

        self.train_acc_metric = pl.metrics.Accuracy()
        self.val_acc_metric = pl.metrics.Accuracy()

        # Populated at test-time
        self.test_metrics: Dict[int, Dict[str, pl.metrics.Metric]] = {}

    @abc.abstractmethod
    def _compute_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the outputs for this model.

        Parameters
        ----------
        inputs
            Inputs to each root module in the batch.
            Has shape [batch_size, root_function_arity, self.hparams.d_model].

        Returns
        -------
        torch.Tensor
            Whatever output this model produces.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_function(self, function_name: str, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the activation for an internal node of the tree.

        Parameters
        ----------
        function_name
            Name of the function to apply.
        inputs
            Inputs to each module in the current step.
            Has shape [batch_size, function_arity, self.hparams.d_model],
            where `batch_size` is the number of nodes being computed at this step.

        Returns
        -------
        torch.Tensor
            The vector to pass up the tree.
        """
        raise NotImplementedError

    @staticmethod
    def _compute_predecessors(
        node_group: torch.Tensor, predecessor_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Given a batch of nodes indices for the same function and an index
        mapping nodes to their predecessors (with placeholders), return just
        the predecessors (without placeholders).

        Parameters
        ----------
        node_group
            A batch of nodes [shape: num_nodes], all of which must have the same arity.
        predecessor_index
            An index (shape: [num_all_nodes, max_in_degree]) mapping nodes to their
            predecessors, which may have placeholders.
        """
        preds_with_placeholders = predecessor_index[node_group]

        # Clip off the placeholders, assuming every row has the same number of them
        arity = (preds_with_placeholders[0] != data.INDEX_PLACEHOLDER).sum().item()
        predecessors = preds_with_placeholders.narrow(dim=1, start=0, length=arity)

        return predecessors

    def forward(self, forest: dgl.DGLGraph) -> torch.Tensor:
        """
        Given a (possibly batched) forest, compute the outputs for each tree.
        """
        num_nodes = forest.num_nodes()
        leaf_mask = (
            forest.ndata[data.FUNCTION_FIELD]
            == self.hparams.function_vocab[data.LEAF_FUNCTION]
        )
        root_mask = (
            forest.ndata[data.FUNCTION_FIELD]
            == self.hparams.function_vocab[data.EQUALITY_FUNCTION]
        )
        internal_mask = ~(leaf_mask | root_mask)
        internal_order = data.typed_topological_nodes_generator(
            forest, node_mask=internal_mask
        )
        predecessor_index = data.tensorize_predecessors(forest)

        # A buffer where the i-th row is the activations output from the i-th node
        # The buffer is repeatedly summed into to allow dense gradient computation;
        # this is valid because each position is summed to exactly once.
        activations = torch.zeros(
            (num_nodes, self.hparams.d_model), device=forest.device
        )

        # Precompute all leaf nodes at once
        tokens = forest.ndata[data.TOKEN_FIELD][leaf_mask]
        token_activations = self.token_embedding(tokens)
        activations = activations.masked_scatter(
            leaf_mask.unsqueeze(1), token_activations
        )

        # Compute all internal nodes under the topological order
        for node_group in internal_order:
            # Look up the type of the first node, since they should all be the same
            function_idx = forest.ndata[data.FUNCTION_FIELD][node_group[0]].item()
            function = self.hparams.function_vocab_inverse[function_idx]

            # Gather inputs, call the module polymorphically, and scatter to buffer
            predecessors = self._compute_predecessors(node_group, predecessor_index)
            inputs = activations[predecessors]
            step_activations = self._apply_function(function, inputs)
            activation_scatter = torch_scatter.scatter(
                src=step_activations, index=node_group, dim=0, dim_size=num_nodes
            )
            activations = activations + activation_scatter

        # Compute all root nodes in 1 step
        root_nodes = torch.where(root_mask)[0]
        root_predecessors = self._compute_predecessors(root_nodes, predecessor_index)
        root_input = activations[root_predecessors]
        return self._compute_output(root_input)

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

    @staticmethod
    def _make_test_metrics():
        return {
            "accuracy": pl.metrics.classification.Accuracy(),
            "precision": pl.metrics.classification.Precision(),
            "recall": pl.metrics.classification.Recall(),
            "f1": pl.metrics.classification.Fbeta(beta=1.0),
        }

    def test_step(self, batch, batch_idx) -> None:
        forest, labels = batch
        logits = self(forest)

        for tree, logit, label in zip(dgl.unbatch(forest), logits, labels):
            logit = logit.cpu()
            label = label.cpu()
            depth = data.tree_depth(tree)

            if depth not in self.test_metrics.keys():
                self.test_metrics[depth] = self._make_test_metrics()

            for metric in self.test_metrics[depth].values():
                metric(logit, label)

    def test_epoch_end(self, _) -> None:
        for depth, depth_metrics in self.test_metrics.items():
            for name, metric in depth_metrics.items():
                self.log(f"{name}/{depth}", metric.compute())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class TreeRNN(RecursiveNN):
    """
    A TreeRNN model.

    For full parameters, see the docstring for `RecursiveNN`.

    Parameters
    ----------
    num_module_layers
        How many layers to use for each internal module.
    """

    def __init__(
        self,
        d_model: int,
        num_module_layers: int,
        learning_rate: float,
        function_vocab: Dict[str, int],
        token_vocab: Dict[str, int],
    ):
        self.save_hyperparameters()
        super().__init__(d_model, learning_rate, function_vocab, token_vocab)

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

    def _compute_output(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = (inputs[:, 0, :] * inputs[:, 1, :]).sum(-1) + self.output_bias
        return logits

    def _apply_function(self, function_name: str, inputs: torch.Tensor) -> torch.Tensor:
        if function_name in data.UNARY_FUNCTIONS:
            module = self.unary_function_modules[function_name]
            inputs = inputs[:, 0, :]
            return module(inputs)

        if function_name in data.BINARY_FUNCTIONS:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            inputs_together = inputs.view(inputs.size(0), -1)
            return module(inputs_together)

        assert False
