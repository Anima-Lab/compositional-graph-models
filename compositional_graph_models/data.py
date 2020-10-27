"""
Code to load equation data in a unified way, as DGLGraphs.

Each example is a directed DGLGraph, with the following int64 node data fields:
 - FUNCTION_FIELD: Indicates the type of function applied at this node. Leaf nodes
        (which represent atoms) have function type LEAF_FUNCTION.
 - TOKEN_FIELD: For leaf nodes, which token is present at that index.
        Function nodes (internal nodes) have value NONLEAF_TOKEN.

Graphs are batched by combining them into one larger disconnected graph via dgl.batch().
This enables greater parallelism when traversing the nodes by type.
"""

import itertools as it
import json
import logging
from typing import Dict, List, Optional, Set, Tuple

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import pytorch_lightning as pl
import torch


FUNCTION_FIELD = "function"
TOKEN_FIELD = "token"

LEAF_FUNCTION = "<Atom>"
UNARY_FUNCTIONS = {"cos", "csc", "tan", "sec", "cot", "sin"}
BINARY_FUNCTIONS = {"Mul", "Add", "Pow"}
EQUALITY_FUNCTION = "Equality"

NONLEAF_TOKEN = "<Function>"

INDEX_PLACEHOLDER = -1

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def build_vocabs(serialized_equations) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build a function vocabulary and a variable vocabulary from a serialized dataset.

    Parameters
    ----------
    serialized_equations
        A list of equations in the serialization format of the equation verification
        task.

    Returns
    -------
    function_vocab : Dict[str, int]
        A vocabulary of function symbols.
    token_vocab : Dict[str, int]
        A vocabulary of variable and number symbols.
    """
    functions: Set[str] = {LEAF_FUNCTION}
    tokens: Set[str] = {NONLEAF_TOKEN}

    for ex in serialized_equations:
        eq = ex["equation"]
        eq_funcs = eq["func"].split(",")
        eq_tokens = eq["vars"].split(",")
        for func, token in zip(eq_funcs, eq_tokens):
            if token:
                tokens.add(token)
            elif func != "#":
                functions.add(func)

    function_vocab = {sym: i for i, sym in enumerate(functions)}
    token_vocab = {sym: i for i, sym in enumerate(tokens)}
    return function_vocab, token_vocab


def _deserialize_binary_recurse(
    func: List[str],
    vars: List[str],
    function_vocab: Dict[str, int],
    token_vocab: Dict[str, int],
) -> dgl.DGLGraph:
    """
    Recursive helper function for `deserialize_binary`.

    Do not use this function directly; call `deserialize_binary` instead.

    Parameters
    ----------
    func
        Function symbols in the tree's preorder traversal. Modified in place.
    vars
        Variable symbols in the tree's preorder traversal. Modified in place.
    function_vocab
        Mapping from function names to indices.
    token_vocab
        Mapping from token names to indices.

    Returns
    -------
    DGLGraph
        A homogeneous graph representing the parseable prefix of the serialized tree.
    """
    if not func:
        raise ValueError(
            "Cannot deserialize an empty structure. Empty trees are serialized as '#'."
        )
    if len(func) != len(vars):
        raise ValueError("`func` and `vars` must have the same length.")

    node_function = func.pop(0)
    node_token = vars.pop(0)

    if node_function == "#":  # Base case: empty graph
        return dgl.graph(([], []))

    if node_token:
        node_function = LEAF_FUNCTION
    else:
        node_token = NONLEAF_TOKEN

    function_id = function_vocab[node_function]
    token_id = token_vocab[node_token]

    # Recursively collect left and right children
    left_subgraph = _deserialize_binary_recurse(func, vars, function_vocab, token_vocab)
    right_subgraph = _deserialize_binary_recurse(
        func, vars, function_vocab, token_vocab
    )

    left_root_idx = left_subgraph.num_nodes() - 1
    right_root_idx = left_root_idx + right_subgraph.num_nodes()
    root_idx = right_root_idx + 1

    graph = dgl.batch([left_subgraph, right_subgraph])
    graph.add_nodes(1)

    if root_idx == 0:  # Leaf node
        graph.ndata[FUNCTION_FIELD] = torch.tensor([function_id], dtype=torch.int64)
        graph.ndata[TOKEN_FIELD] = torch.tensor([token_id], dtype=torch.int64)
    else:
        if left_subgraph.num_nodes() == 0:  # Right-unary
            graph.add_edges(right_root_idx, root_idx)
        elif right_subgraph.num_nodes() == 0:  # Left-unary
            graph.add_edges(left_root_idx, root_idx)
        else:  # Binary
            graph.add_edges([left_root_idx, right_root_idx], root_idx)

        graph.ndata[FUNCTION_FIELD][-1] = function_id
        graph.ndata[TOKEN_FIELD][-1] = token_id

    return graph


def deserialize_binary(
    equation: Dict[str, str],
    function_vocab: Dict[str, int],
    token_vocab: Dict[str, int],
) -> dgl.DGLGraph:
    """
    Deserialize binary equation data from the "equation_verification_40k" format
    into a DGLGraph. See the module docstring for the graph properties.

    Parameters
    ----------
    equation
        The "equation" field of a serialized example; a serialized binary tree.
    function_vocab
        A mapping from function name to id. See `build_vocabs()`.
    token_vocab
        A mapping from token name to id. See `build_vocabs()`.

    Returns
    -------
    DGLGraph
        The equation tree, deserialized into a usable data structure.
    """
    func = equation["func"].split(",")
    vars = equation["vars"].split(",")
    return _deserialize_binary_recurse(func, vars, function_vocab, token_vocab)


def plot_graph(
    graph: dgl.DGLGraph, function_vocab: Dict[str, int], token_vocab: Dict[str, int]
) -> None:
    """
    Plot an equation graph in a human-readable format.

    Each node is labeled with its token type if it is a leaf or its function type if it
    is internal, as well as its input index.

    Parameters
    ----------
    graph
        The graph to display.
    function_vocab
        A mapping from function name to id.
    token_vocab
        A mapping from token name to id.
    """
    function_names = {idx: name for name, idx in function_vocab.items()}
    token_names = {idx: name for name, idx in token_vocab.items()}

    labels = {}
    for i, (function_id, token_id) in enumerate(
        zip(graph.ndata[FUNCTION_FIELD].numpy(), graph.ndata[TOKEN_FIELD].numpy(),)
    ):
        if function_names[function_id] == LEAF_FUNCTION:
            labels[i] = f"{token_names[token_id]}"
        else:
            labels[i] = f"{function_names[function_id]}"

    nx_graph = graph.to_networkx()
    positions = nx.nx_agraph.graphviz_layout(nx_graph, prog="dot")
    nx.draw(nx_graph, positions, with_labels=False)
    nx.draw_networkx_labels(nx_graph, positions, labels)

    plt.axis("off")
    plt.show(block=False)


def typed_topological_nodes_generator(
    graph: dgl.DGLGraph, node_mask: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    """
    Generates a topological traversal for the nodes of `graph`, guaranteeing the
    following two properties:
        - A node is visited only if all of its descendants have been visited.
        - At each step, all nodes visited have the same function type (FUNCTION_FIELD).

    Additionally, the algorithm will heuristically attempt to maximize parallelism by
    greedily selecting the node type with the most available nodes at each step.

    Parameters
    ----------
    graph
        The graph whose nodes to traverse. May have multiple independent graphs
        packaged together with `dgl.batch()`.
    node_mask
        If provided, any node with a 0 in this mask is ignored.
        Must be a bool tensor or None.

    Returns
    -------
    List[Tensor]
        The i-th element in this list is a tensor consisting of the nodes to visit
        at step i. The union of these lists is exactly the node indices of `graph`.
    """
    visited = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=graph.device)
    remaining_dependencies = graph.in_degrees()
    order: List[torch.Tensor] = []

    if node_mask is not None:
        opposite_mask = ~node_mask
        visited |= opposite_mask
        remaining_dependencies *= node_mask
        (opposite_indices,) = torch.nonzero(opposite_mask, as_tuple=True)
        _, masked_successors = graph.out_edges(opposite_indices)
        visit_counts = torch.bincount(masked_successors, minlength=graph.num_nodes())
        remaining_dependencies -= visit_counts

    while not visited.all():
        # Find unvisited nodes with no unfulfilled dependencies
        available = visited.bitwise_not() & (remaining_dependencies == 0)

        # Greedily select the type with the most available nodes
        available_types = graph.ndata[FUNCTION_FIELD].masked_select(available)
        step_type, _ = available_types.mode()

        # Find all nodes of this type with no additional dependencies
        step_mask = available & (graph.ndata[FUNCTION_FIELD] == step_type)

        # Flag these nodes as visited, convert the mask to indices, and add to the order
        visited |= step_mask
        (step_indices,) = torch.nonzero(step_mask, as_tuple=True)
        order.append(step_indices)

        # Tick down the dependency count of these nodes' successors
        _, successor_indices = graph.out_edges(step_indices)
        visit_counts = torch.bincount(successor_indices, minlength=graph.num_nodes())
        remaining_dependencies -= visit_counts

    return order


def tensorize_predecessors(graph: dgl.DGLGraph) -> torch.Tensor:
    """
    Given a DAG G, produce a tensor `index` of shape [N, D] where:
        - N is the number of nodes in G
        - D is the maximum in-degree in G
        - index[n, i] is the i-th predecessor of node n, or -1 if none exists.

    Note that this should work for non-binary graphs.

    Parameters
    ----------
    graph
        A directed acyclic graph, as a homogeneous DGLGraph.

    Returns
    -------
    torch.Tensor
        The tensor `index` described above.
    """
    src, dst = graph.edges()

    # Assume edges are in increasing order; count the number of children for each node
    dst_unique, dst_count = dst.unique_consecutive(return_counts=True)
    dst_count = dst_count.cpu()

    # Partition the source nodes according to their destination node;
    # produces a list of tensors, where the i-th tensor is the children of node i
    src_partition = src.split(list(dst_count))

    # Create a dense tensor from src_partition with a placeholder for any node that
    # doesn't have a full set of children. This is `index` without any degree-0 nodes.
    partition_padded = torch.nn.utils.rnn.pad_sequence(
        src_partition, padding_value=INDEX_PLACEHOLDER, batch_first=True
    )

    max_in_degree = dst_count.max().item()
    index = torch.full(
        (graph.num_nodes(), max_in_degree),
        INDEX_PLACEHOLDER,
        dtype=torch.int64,
        device=graph.device,
    )

    # Write any rows for nodes having children into the index
    index[dst_unique] = partition_padded

    return index


class _EquationTreeDataset(torch.utils.data.Dataset):
    """
    A Dataset of equation trees and their labels.

    Each batch is a (graph, label) pair, where `graph` is a (batched) DGLGraph in the
    format described in the module docstring and `label` is a batch of float labels.

    Parameters
    ----------
    raw_data
        A flat list of dictionaries, each with "equation" and "label" keys in the
        "equation_verification_40k" format.
    function_vocab
        Mapping from function names to indices.
    token_vocab
        Mapping from token names to indices.
    """

    def __init__(
        self, raw_data, function_vocab: Dict[str, int], token_vocab: Dict[str, int]
    ):
        self._graphs: List[dgl.DGLGraph] = []
        self._labels: List[torch.Tensor] = []

        for datapoint in raw_data:
            graph = deserialize_binary(
                datapoint["equation"], function_vocab, token_vocab
            )
            label = torch.tensor(float(datapoint["label"]))
            self._graphs.append(graph)
            self._labels.append(label)

        _logger.info(f"Deserialized {len(self)} graphs.")

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        return (self._graphs[idx], self._labels[idx])

    def __len__(self) -> int:
        return len(self._labels)

    @staticmethod
    def collate_batch(
        batch: List[Tuple[dgl.DGLGraph, torch.Tensor]]
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        graphs, labels = zip(*batch)
        batched_graphs = dgl.batch(graphs)
        batched_labels = torch.stack(labels)
        return batched_graphs, batched_labels


class EquationTreeDataModule(pl.LightningDataModule):
    """
    A DataModule which loads training, validation, and testing data (any of which are
    optional) and creates a vocabulary that will be valid across the datasets.

    Note that the vocabulary generated is NOT guaranteed to be the same between runs!
    You must save your vocabularies alongside your model checkpoints, and only ever use
    a model with its initial vocabularies. Otherwise the model embeddings will not be
    indexed correctly.

    Parameters
    ----------
    train_path
        Path to the training data to use, if any.
    val_path
        Path to the validation data to use, if any.
    test_path
        Path to the testing data to use, if any.
    batch_size
        Batch size to use for all datasets.
    function_vocab
        If provided, use this as the vocabulary mapping function names to indices.
        Must be provided iff `token_vocab` is.
    token_vocab
        If provided, use this as the vocabulary mapping token names to indices.
        Must be provided iff `function_vocab` is.
    num_data_workers
        Number of processes to use for data loading.
    pin_memory
        Whether to pin GPU memory.
    """

    function_vocab: Dict[str, int]
    token_vocab: Dict[str, int]

    train_dataset: Optional[_EquationTreeDataset] = None
    val_dataset: Optional[_EquationTreeDataset] = None
    test_dataset: Optional[_EquationTreeDataset] = None

    @staticmethod
    def _load_path(path: str):
        with open(path, "r") as f:
            data_raw = json.load(f)

        return list(it.chain.from_iterable(data_raw))

    def __init__(
        self,
        train_path: Optional[str],
        val_path: Optional[str],
        test_path: Optional[str],
        batch_size: int,
        function_vocab: Optional[Dict[str, int]] = None,
        token_vocab: Optional[Dict[str, int]] = None,
        num_data_workers: int = 4,
        pin_memory: bool = True,
    ):
        if not (train_path or val_path or test_path):
            raise ValueError("At least one data path must be specified.")

        if (function_vocab and not token_vocab) or (token_vocab and not function_vocab):
            raise ValueError("Either both vocabularies must be supplied, or neither.")

        if (not function_vocab) and (not train_path):
            _logger.warning(
                "Building a dataset with no vocabulary and validation/test data only. "
                "This WILL NOT share a vocabulary with a pretrained model! "
                "Pass in a pre-existing vocabulary to evaluate or resume training."
            )

        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self._function_vocab = function_vocab
        self._token_vocab = token_vocab

        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        build_vocab = not self._function_vocab
        all_data_raw = []

        if self.train_path:
            train_data_raw = self._load_path(self.train_path)
            _logger.info(f"Loaded {len(train_data_raw)} training examples.")
            if build_vocab:
                all_data_raw.extend(train_data_raw)

        if self.val_path:
            val_data_raw = self._load_path(self.val_path)
            _logger.info(f"Loaded {len(val_data_raw)} validation examples.")
            if build_vocab:
                all_data_raw.extend(val_data_raw)

        if self.test_path:
            test_data_raw = self._load_path(self.test_path)
            _logger.info(f"Loaded {len(test_data_raw)} test examples.")
            if build_vocab:
                all_data_raw.extend(test_data_raw)

        if build_vocab:
            self.function_vocab, self.token_vocab = build_vocabs(all_data_raw)
            _logger.info(
                f"Built vocabularies with {len(self.function_vocab)} functions and "
                f"{len(self.token_vocab)} tokens."
            )
            del all_data_raw
        else:
            assert self._function_vocab and self._token_vocab
            self.function_vocab = self._function_vocab
            self.token_vocab = self._token_vocab
            _logger.info(
                f"Loaded vocabularies with {len(self.function_vocab)} functions and "
                f"{len(self.token_vocab)} tokens."
            )

        if self.train_path:
            _logger.info("Loading training set...")
            self.train_dataset = _EquationTreeDataset(
                train_data_raw, self.function_vocab, self.token_vocab
            )
            del train_data_raw

        if self.val_path:
            _logger.info("Loading validation set...")
            self.val_dataset = _EquationTreeDataset(
                val_data_raw, self.function_vocab, self.token_vocab
            )
            del val_data_raw

        if self.test_path:
            _logger.info("Loading test set...")
            self.test_dataset = _EquationTreeDataset(
                test_data_raw, self.function_vocab, self.token_vocab
            )
            del test_data_raw

    def train_dataloader(self):
        assert self.train_path
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_EquationTreeDataset.collate_batch,
            num_workers=self.num_data_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        assert self.val_path
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_EquationTreeDataset.collate_batch,
            num_workers=self.num_data_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        assert self.test_path
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_EquationTreeDataset.collate_batch,
            num_workers=self.num_data_workers,
            pin_memory=self.pin_memory,
        )

    def transfer_batch_to_device(self, batch, device):
        graph, labels = batch
        graph = graph.to(device)
        labels = labels.to(device)
        return graph, labels
