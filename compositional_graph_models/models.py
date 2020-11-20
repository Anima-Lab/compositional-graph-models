"""
Implements models for the equation verification task.

Models are implemented using DGL. See `data.py` for graph format.
"""

import abc
from typing import Callable, Dict, List, Tuple

import dgl
import dgl.nn
import pytorch_lightning as pl
import torch
import torch_scatter

import data


class FunctionModule(torch.nn.Module):
    """
    An MLP block that takes in a fixed number of embedding vectors, and produces single
    embedding vectors.

    Parameters
    ----------
    arity
        Number of (concatenated) embedding vectors to expect as input.
    d_model
        Dimensionality of the output. The input is expected to be twice this size.
    num_layers
        How many dense layers to apply.
    """

    def __init__(self, arity: int, d_model: int, num_layers: int):
        if arity <= 0:
            raise ValueError("A function module must take at least one input.")
        if num_layers <= 0:
            raise ValueError("A function module must have at least one layer.")

        super().__init__()

        layers: List[torch.nn.Module] = []
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(arity * d_model, arity * d_model))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(arity * d_model, d_model))
        layers.append(torch.nn.Tanh())
        self.layer_stack = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(inputs)


class UnaryLSTM(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.data = torch.nn.Linear(d_model, d_model, bias=True)
        self.forget = torch.nn.Linear(d_model, d_model, bias=True)
        self.output = torch.nn.Linear(d_model, d_model, bias=True)
        self.input = torch.nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor, c: torch.Tensor, dropout=None):
        """
        Computes a forward pass of Unary LSTM node.
        Args:
            h: Hidden state of the children. Dim: [batch_size, d_model]
            c: Cell state of the children. Dim: [batch_size, 1, d_model]

        Returns:
            (hidden, cell): Hidden state and cell state of parent. 
            Hidden dim: [batch_size, d_model]. Cell dim: [batch_size, 1, d_model]
        """
        c = c.squeeze(1)
        i = torch.sigmoid(self.data(h))
        f = torch.sigmoid(self.forget(h))
        o = torch.sigmoid(self.output(h))
        u = torch.tanh(self.input(h))
        if dropout is None:
            cp = i * u + f * c
        else:
            cp = i * F.dropout(u,p=dropout,training=self.training) + f * c
        hp = o * torch.tanh(cp)
        return (hp, cp.unsqueeze(1))


class BinaryLSTM(torch.nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.data_left = torch.nn.Linear(d_model, d_model, bias=False)
        self.data_right = torch.nn.Linear(d_model, d_model, bias=False)
        self.data_bias = torch.nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_left_by_left = torch.nn.Linear(d_model, d_model, bias=False)
        self.forget_left_by_right = torch.nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_left = torch.nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_right = torch.nn.Linear(d_model, d_model, bias=False)
        self.forget_bias_left = torch.nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_bias_right = torch.nn.Parameter(torch.FloatTensor([0] * d_model))
        self.output_left = torch.nn.Linear(d_model, d_model, bias=False)
        self.output_right = torch.nn.Linear(d_model, d_model, bias=False)
        self.output_bias = torch.nn.Parameter(torch.FloatTensor([0] * d_model))
        self.input_left = torch.nn.Linear(d_model, d_model, bias=False)
        self.input_right = torch.nn.Linear(d_model, d_model, bias=False)
        self.input_bias = torch.nn.Parameter(torch.FloatTensor([0] * d_model))

    def forward(self, hl, hr, cl, cr, dropout=None): 
        """
        Computes a forward pass of Binary LSTM node.
        Args:
            hl, hr: Hidden states of the children. Dim: [batch_size, d_model]
            cl, cr: Cell states of the children. Dim: [batch_size, 1, d_model]

        Returns:
            (hp, cp): Hidden state and cell state of parent. 
            Hidden dim: [batch_size, d_model]. Cell dim: [batch_size, 1, d_model]
        """
        cl = cl.squeeze(1)
        cr = cr.squeeze(1)
        i = torch.sigmoid(self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget_left_by_left(hl) + self.forget_left_by_right(hr) + self.forget_bias_left)
        f_right = torch.sigmoid(self.forget_right_by_left(hl) + self.forget_right_by_right(hr) + self.forget_bias_right)
        o = torch.sigmoid(self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = torch.tanh(self.input_left(hl) + self.input_right(hr) + self.input_bias)
        if dropout is None:
            cp = i * u + f_left * cl + f_right * cr
        else:
            cp = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        hp = o * torch.tanh(cp)
        return (hp, cp.unsqueeze(1))



class GraphClassifier(pl.LightningModule, abc.ABC):
    """
    A superclass for models which will be used for binary classification on graphs.

    Parameters
    ----------
    learning_rate
        Learning rate to use for training.
    function_vocab
        Vocabulary mapping function names to indices.
    token_vocab
        Vocabulary mapping token names to indices.
    """

    def __init__(
        self,
        learning_rate: float,
        function_vocab: Dict[str, int],
        token_vocab: Dict[str, int],
    ):
        super().__init__()

        self.hparams.function_vocab_inverse = {v: k for k, v in function_vocab.items()}
        self.hparams.token_vocab_inverse = {v: k for k, v in token_vocab.items()}

        self.train_acc_metric = pl.metrics.Accuracy()
        self.val_acc_metric = pl.metrics.Accuracy()

        # Populated at test-time
        self.test_metrics: Dict[int, Dict[str, pl.metrics.Metric]] = {}

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
        logits = self(forest.clone())  # Cloning enables in-place modification

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

    @abc.abstractmethod
    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Given a DGLGraph (possibly batched), produce one binary logit per batch graph.
        """
        raise NotImplementedError


class RecursiveNN(GraphClassifier):
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
    """

    def __init__(
        self,
        d_model: int,
        learning_rate: float,
        function_vocab: Dict[str, int],
        token_vocab: Dict[str, int],
    ):
        super().__init__(learning_rate, function_vocab, token_vocab)

        self.token_embedding = torch.nn.Embedding(
            num_embeddings=len(token_vocab),
            embedding_dim=d_model,
            padding_idx=token_vocab[data.NONLEAF_TOKEN],
        )

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
    def _apply_function(self, function_name: str, input_cell: torch.Tensor, input_hidden: torch.Tensor) -> torch.Tensor:
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

        # A buffer where the i-th row is the memory output from the i-th node.
        memory = torch.zeros(
            (num_nodes, self.memory_size, self.hparams.d_model), device=forest.device
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
            input_cell = activations[predecessors]
            input_hidden = memory[predecessors]
            step_activations, step_memory = self._apply_function(function, input_cell, input_hidden)
            activation_scatter = torch_scatter.scatter(
                src=step_activations, index=node_group, dim=0, dim_size=num_nodes
            )
            memory_scatter = torch_scatter.scatter(
                src=step_memory, index=node_group, dim=0, dim_size=num_nodes
            )
            activations = activations + activation_scatter
            memory = memory + memory_scatter

        # Compute all root nodes in 1 step
        root_nodes = torch.where(root_mask)[0]
        root_predecessors = self._compute_predecessors(root_nodes, predecessor_index)
        root_input = activations[root_predecessors]
        return self._compute_output(root_input)

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
                f: FunctionModule(1, d_model, num_module_layers)
                for f in data.UNARY_FUNCTIONS
            }
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {
                f: FunctionModule(2, d_model, num_module_layers)
                for f in data.BINARY_FUNCTIONS
            }
        )
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))
        self.memory_size = 1

    def _compute_output(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = (inputs[:, 0, :] * inputs[:, 1, :]).sum(-1) + self.output_bias
        return logits

    def _apply_function(self, function_name: str, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if function_name in data.UNARY_FUNCTIONS:
            module = self.unary_function_modules[function_name]
            inputs = inputs[:, 0, :]
            return module(inputs), memory[:, 0, :]

        if function_name in data.BINARY_FUNCTIONS:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            inputs_together = inputs.view(inputs.size(0), -1)
            return module(inputs_together), memory[:, 0, :]

        assert False


class TreeLSTM(RecursiveNN):
    """
    A TreeLSTM model.

    For full parameters, see the docstring for `RecursiveNN`.

    Parameters
    ----------
    num_module_layers
        How many layers to use for each internal module.
    """

    def __init__(
        self,
        d_model: int,
        learning_rate: float,
        function_vocab: Dict[str, int],
        token_vocab: Dict[str, int],
    ):
        self.save_hyperparameters()
        super().__init__(d_model, learning_rate, function_vocab, token_vocab)

        self.unary_function_modules = torch.nn.ModuleDict(
            {
                f: UnaryLSTM(d_model)
                for f in data.UNARY_FUNCTIONS
            }
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {
                f: BinaryLSTM(d_model)
                for f in data.BINARY_FUNCTIONS
            }
        )
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))
        self.memory_size = 1

    def _compute_output(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = (inputs[:, 0, :] * inputs[:, 1, :]).sum(-1) + self.output_bias
        return logits

    def _apply_function(self, function_name: str, inputs: torch.Tensor, memory: torch.tensor) -> torch.Tensor:
        if function_name in data.UNARY_FUNCTIONS:
            module = self.unary_function_modules[function_name]
            inputs = inputs[:, 0, :]
            memory = memory[:, 0, :]
            return module(inputs, memory)

        if function_name in data.BINARY_FUNCTIONS:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            l_inputs = inputs[:, 0, :]
            r_inputs = inputs[:, 1, :]
            l_memory = memory[:, 0, :]
            r_memory = memory[:, 1, :]
            return module(l_inputs, r_inputs, l_memory, r_memory)

        assert False



class GraphCNN(GraphClassifier):
    """
    A graph convolutional model for equation verification.

    First, every node gets a feature from an embedding lookup on its function or token.
    Then, for each layer, nodes update their features by applying a module
    (shared across all nodes) to four inputs concatenated in order:
        - The left child's features
        - The right child's features
        - The parent's features
        - The node's own features
    If any of these are not present, zeroes are used instead.

    Each expression (left or right of an Equality) is processed separately.
    Then, the expression is represented by its mean feature vector, and the logit
    of the equation is the dot product of the left and right mean vectors.


    Parameters
    ----------
    d_model
        Dimension of the embedding vectors and features.
    num_layers
        Number of times to perform message-passing between nodes.
    num_module_layers
        Number of layers to use in each module for message aggregation.

    See GraphClassifier for other parameter descriptions.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_module_layers: int,
        learning_rate: float,
        function_vocab: Dict[str, int],
        token_vocab: Dict[str, int],
    ):
        self.save_hyperparameters()
        super().__init__(learning_rate, function_vocab, token_vocab)

        # A single vocab for both functions and tokens
        self.hparams.min_function_idx = len(token_vocab)
        self.hparams.combined_vocab = {
            **token_vocab,
            **{
                name: idx + self.hparams.min_function_idx
                for name, idx in function_vocab.items()
            },
        }
        self.hparams.combined_vocab_inverse = {
            v: k for k, v in self.hparams.combined_vocab.items()
        }

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.hparams.combined_vocab),
            embedding_dim=d_model,
            padding_idx=self.hparams.min_function_idx
            + self.hparams.function_vocab[data.PAD_FUNCTION],
        )

        self.layers = torch.nn.ModuleList(
            [FunctionModule(4, d_model, num_module_layers) for _ in range(num_layers)]
        )

        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))

    @staticmethod
    def _reduce_layer(
        layer: torch.nn.Module,
    ) -> Callable[[dgl.udf.NodeBatch], Dict[str, torch.Tensor]]:
        """
        Make a DGL user-defined reduction function from a module.

        The function expects a node batch with features of shape
        [n_nodes, n_arguments, dim], and applies a module which is expected to accept
        tensors of shape n_arguments * dim to produce new features of shape dim.

        Parameters
        ----------
        layer
            The module to apply during reduction. Must be polymorphic in its first
            tensor dimension (i.e. batch-size agnostic).

        Returns
        -------
        NodeBatch -> Dict[str, Tensor]
            A user-defined reduction function which applies `layer` to the flattened
            input messages.
        """

        def layer_udf(nodes: dgl.udf.NodeBatch) -> Dict[str, torch.Tensor]:
            batch_size = nodes.mailbox["features"].size(0)
            features = nodes.mailbox["features"].view(batch_size, -1)
            outputs = layer(features)
            return {"features": outputs}

        return layer_udf

    def _prepare_graph(self, graph: dgl.DGLGraph):
        """
        Transform the input graph inplace into a graph that can be used for computation.

        To process all nodes simultaneously, we add a "padding" node that acts as a
        placeholder child or parent for every node missing one of those, so that every
        node has the same in-degree (4). The padding node always has a feature vector
        of zero.

        Edges are always indexed in the order (left child, right child, parent, self).

        The following invariants hold for all nodes:
            - Leaf nodes have 2 child edges from the padding node.
            - Unary function nodes have a left edge from their child and a right edge
              from the padding node.
            - Expression roots (nodes directly under Equality nodes) have a parent
              edge from the padding node, and there are no Equality nodes.

        Parameters
        ----------
        graph
            The graph to modify. Modified in place; if this is not desirable,
            clone the graph ahead of time and pass in the clone to modify in place.
        """
        # Remove Equality nodes, splitting examples into left and right expressions
        root_mask = (
            graph.ndata[data.FUNCTION_FIELD]
            == self.hparams.function_vocab[data.EQUALITY_FUNCTION]
        )
        root_indices = torch.nonzero(root_mask, as_tuple=False)[:, 0]
        graph.remove_nodes(root_indices)

        # Save the parent -> child edges for later
        parent_graph = graph.reverse()

        # Add a "padding node" to signal missing values
        graph.add_nodes(1)
        pad_node_id = graph.num_nodes() - 1
        graph.ndata["function"][pad_node_id] = self.hparams.function_vocab[
            data.PAD_FUNCTION
        ]
        graph.ndata["token"][pad_node_id] = self.hparams.token_vocab[data.PAD_TOKEN]

        # Add edges from the padding node to all nodes with less than 2 children
        pad_count = 2 - graph.in_degrees()  # Number of times to pad each node
        pad_count[pad_node_id] = 0  # Never add input edges to the pad node

        while not (pad_count == 0).all():
            pad_mask = pad_count > 0
            pad_idx = torch.nonzero(pad_mask, as_tuple=False)[:, 0]
            graph.add_edges(pad_node_id, pad_idx)
            pad_count -= pad_mask.to(pad_count.dtype)

        # Add the parent -> child edges back in
        graph.add_edges(*parent_graph.edges())

        # Add edges from the pad node to every node without a parent, except itself
        pad_idx = torch.nonzero(graph.in_degrees() < 3, as_tuple=False)[:-1, 0]
        graph.add_edges(pad_node_id, pad_idx)

        # Add self-loops to all nodes except the pad node
        internal_indices = graph.nodes()[:-1]
        graph.add_edges(internal_indices, internal_indices)

    def _embed_nodes(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Compute a feature vector for each node in the graph through embedding lookup.

        Each node is embedded based on its token (if it is a leaf) or function
        (if it is not).

        Parameters
        ----------
        graph
            The graph to compute features for.

        Returns
        -------
        Tensor
            A tensor of shape [num_nodes, self.hparams.d_model] containing per-node
            features.
        """
        leaf_mask = (
            graph.ndata[data.FUNCTION_FIELD]
            == self.hparams.function_vocab[data.LEAF_FUNCTION]
        )
        combined_idx = torch.where(
            leaf_mask,
            graph.ndata[data.TOKEN_FIELD],
            graph.ndata[data.FUNCTION_FIELD] + self.hparams.min_function_idx,
        )
        node_embeddings = self.embedding(combined_idx)
        return node_embeddings

    @staticmethod
    def _left_right_subgraphs(graph: dgl.DGLGraph) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
        """
        Given a graph containing multiple equations, produce subgraphs containing just
        the left (resp. the right) expressions in those equalities.

        Works with both a typical input graph and a computational graph processed
        through _prepare_graph().

        Parameters
        ----------
        graph
            A graph containing multiple equalities.

        Returns
        -------
        left_subgraph : DGLGraph
            A graph of expressions on the left-hand side of equations in the input.
        right_subgraph : DGLGraph
            A graph of expressions on the right-hand side of equations in the input.
        """
        left_subgraph = dgl.node_subgraph(
            graph, graph.ndata[data.SIDE_FIELD] == data.SIDE_LEFT
        )
        right_subgraph = dgl.node_subgraph(
            graph, graph.ndata[data.SIDE_FIELD] == data.SIDE_RIGHT
        )

        # Workaround for https://github.com/dmlc/dgl/issues/2310.
        # Equations list all of their left nodes before all of their right nodes, so
        # count the nodes on each side by running sum, then take alternating elements.
        subexpression_node_counts = torch.unique_consecutive(
            graph.ndata[data.SIDE_FIELD], return_counts=True
        )[1]
        left_counts = subexpression_node_counts[::2]
        right_counts = subexpression_node_counts[1::2]

        left_subgraph.set_batch_num_nodes(left_counts)
        left_subgraph.set_batch_num_edges(left_counts - 1)

        right_subgraph.set_batch_num_nodes(right_counts)
        right_subgraph.set_batch_num_edges(right_counts - 1)

        return left_subgraph, right_subgraph

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Compute the logits of equality holding for each equation in the (batched) graph.

        NOTE: modifies the graph in place.

        Parameters
        ----------
        graph
            The graph (possibly batched) containing equalities to classify.

        Returns
        -------
        Tensor
            For each equality in the input graph, the logit of positive classification.
        """
        self._prepare_graph(graph)
        graph.ndata["features"] = self._embed_nodes(graph)

        for layer in self.layers:
            reduce_udf = self._reduce_layer(layer)
            graph.update_all(
                message_func=dgl.function.copy_src("features", "features"),
                reduce_func=reduce_udf,
            )

        pad_node_id = graph.num_nodes() - 1
        graph.remove_nodes(pad_node_id)
        left_subgraph, right_subgraph = self._left_right_subgraphs(graph)

        # TODO: try other readouts
        encoding_left = dgl.readout_nodes(left_subgraph, "features", op="mean")
        encoding_right = dgl.readout_nodes(right_subgraph, "features", op="mean")

        # TODO: try other metrics like cosine similarity
        logits = (encoding_left * encoding_right).sum(-1) + self.output_bias
        return logits
