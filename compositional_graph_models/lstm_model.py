class UnaryLSTM(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.data = nn.Linear(d_model, d_model, bias=True)
        self.forget = nn.Linear(d_model, d_model, bias=True)
        self.output = nn.Linear(d_model, d_model, bias=True)
        self.input = nn.Linear(d_model, d_model, bias=True)

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
        self.data_left = nn.Linear(d_model, d_model, bias=False)
        self.data_right = nn.Linear(d_model, d_model, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_left_by_left = nn.Linear(d_model, d_model, bias=False)
        self.forget_left_by_right = nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_left = nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_right = nn.Linear(d_model, d_model, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.output_left = nn.Linear(d_model, d_model, bias=False)
        self.output_right = nn.Linear(d_model, d_model, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.input_left = nn.Linear(d_model, d_model, bias=False)
        self.input_right = nn.Linear(d_model, d_model, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * d_model))

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

        # A buffer where the i-th row is the memory output from the i-th node.
        memory = torch.zeros(
            (num_nodes, self.hparams.memory_size, self.hparams.d_model), device=forest.device
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
        num_module_layers: int,
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
