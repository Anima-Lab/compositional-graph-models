import torch 
import torch.nn.functional as F

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


class UnarySMU(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        if params.top_k == 1 and params.gate_top_k:
            raise AssertionError('gating top-k is only supported with top_k>1')
        
        self.d_model = params.d_model
        self.o_linear = torch.nn.Linear(self.d_model, self.d_model, bias=True)
        self.activation = torch.sigmoid if params.tree_activation == "sigmoid" else torch.tanh
        self.top_k = params.top_k
        self.gate_top_k = params.gate_top_k
        

        if self.top_k > 1:
            if not self.gate_top_k:
                self.pos_linear = torch.nn.Linear(self.d_model, self.top_k, bias=True)
            else:
                for k in range(1,self.top_k+1):
                    setattr(self, 'pos_linear_'+str(k), torch.nn.Linear(self.d_model, self.d_model, bias=True))

    def forward(self, inp, node_memory):
        """
        Args:
            inp: (batch_size, d_model)
            memory: (batch_size, stack_size, d_model)

        Returns:
            (batch_size, d_model) 
        """
        batch_size = inp.size()[0]
        o = self.o_linear(inp)
        o = torch.sigmoid(o) # (batch_size, d_model)
        if self.top_k > 1: 
            if not self.gate_top_k:
                pos_gate = self.pos_linear(inp)
                pos_gate = torch.sigmoid(pos_gate) # (batch_size, top_k)
                top_indices = torch.tensor(range(0, self.top_k), device=inp.device)
                memory = torch.index_select(node_memory, 1, top_indices) # (batch_size, top_k, d_model)
                memory = torch.bmm(pos_gate.unsqueeze(1), memory)
            else:
                memory = torch.zeros(
                    (batch_size, 1, self.d_model), dtype=torch.float, device=inp.device
                )
                for k in range(1,self.top_k+1):
                    # IS THIS GOING TO BE REALLY SLOW?
                    nn_block = getattr(self, 'pos_linear_{0}'.format(k))
                    pos_gate = nn_block(inp)
                    pos_gate = torch.sigmoid(pos_gate)
                    mem_row = torch.narrow(node_memory, 1, k-1, 1)
                    memory = memory + pos_gate.unsqueeze(1) * mem_row
        else:
            memory = torch.narrow(node_memory, 1, 0, 1)

        output = self.activation(memory)
        output = o.unsqueeze(1) * output

        return output


class BinarySMU(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        if params.top_k == 1 and params.gate_top_k:
            raise AssertionError('gating top-k is only supported with top_k>1')

        self.d_model = params.d_model
        self.o_linear = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)
        self.activation = torch.sigmoid if params.tree_activation == "sigmoid" else torch.tanh
        self.top_k = params.top_k
        self.gate_top_k = params.gate_top_k

        if params.top_k > 1:
            if not self.gate_top_k:
                self.pos_linear = torch.nn.Linear(self.d_model * 2, params.top_k, bias=True)
            else:
                for k in range(1,self.top_k+1):
                    setattr(self, 'pos_linear_'+str(k), torch.nn.Linear(self.d_model * 2, self.d_model, bias=True))

    def forward(self, input_left, input_right, node_memory):
        
        inp = torch.cat((input_left, input_right), dim=1)
        batch_size = inp.size()[0]

        o = self.o_linear(inp)
        o = torch.sigmoid(o) # (batch_size, 2 * d_model)
        if self.top_k > 1: 
            if not self.gate_top_k:
                pos_gate = self.pos_linear(inp)
                pos_gate = torch.sigmoid(pos_gate) # (batch_size, top_k)
                top_indices = torch.tensor(range(0, self.top_k), device=inp.device)
                memory = torch.index_select(node_memory, 1, top_indices) # (batch_size, top_k, d_model)
                memory = torch.bmm(pos_gate.unsqueeze(1), memory)
            else:
                memory = torch.zeros((batch_size,1,self.num_input), dtype=torch.float, requires_grad=False, device=inp.device)
                for k in range(1,self.top_k+1):
                    # IS THIS GOING TO BE REALLY SLOW?
                    nn_block = getattr(self, 'pos_linear_{0}'.format(k))
                    pos_gate = nn_block(inp)
                    pos_gate = torch.sigmoid(pos_gate)
                    mem_row = torch.narrow(node_memory, 1, k-1, 1)
                    memory = memory + pos_gate.unsqueeze(1) * mem_row
        else:
            memory = torch.narrow(node_memory, 1, 0, 1)

        output = self.activation(memory)
        output = o.unsqueeze(1) * output

        return output


class UnaryStack(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        assert not (params.no_op and params.no_pop) # at least one of these should be False

        self.d_model    = params.d_model
        self.stack_activations = torch.sigmoid if params.stack_activation == "sigmoid" else torch.tanh

        if params.gate_push_pop:
            self.push_gate_linear = torch.nn.Linear(self.d_model, self.d_model, bias=True)
            if not params.no_pop:
                self.pop_gate_linear = torch.nn.Linear(self.d_model, self.d_model, bias=True)
            if params.no_op or params.no_pop:
                self.no_op_gate_linear = torch.nn.Linear(self.d_model, self.d_model, bias=True)
        else:
            if params.no_op:
                self.action = torch.nn.Linear(self.d_model, 3, bias=True)
            else:
                self.action = torch.nn.Linear(self.d_model, 2, bias=True)

        self.gate_linear = torch.nn.Linear(self.d_model, self.d_model, bias=True)
        self.input_linear = torch.nn.Linear(self.d_model, self.d_model, bias=True)
        if params.like_LSTM:
            self.data_linear = torch.nn.Linear(self.d_model, self.d_model, bias=True)

        self.stack_size = params.stack_size
        self.no_op      = params.no_op
        self.no_pop = params.no_pop
        self.like_LSTM = params.like_LSTM
        self.gate_push_pop = params.gate_push_pop
        self.normalize_action = params.normalize_action

    def forward(self, inp, stack, dropout=None):
        """
        Args:
            inp: (batch_size, d_model)
            stack: (batch_size, stack_size, d_model)

        Returns:
            (batch_size, stack_size, d_model)
        """
        batch_size = inp.size()[0]
        inp = inp.unsqueeze(1)

        # Push zeros onto stack as placeholder
        gate = self.gate_linear(inp)
        gate = torch.sigmoid(gate) # (batch_size, 1)
        stack = gate * stack
        zeros = torch.zeros(
                    (batch_size, 1, self.d_model), dtype=torch.float, requires_grad=True, device=inp.device
                )
        stack = torch.cat([stack, zeros], dim=1)

        # Push
        push_input = self.input_linear(inp) # (batch_size, 1, d_model)
        push_input = self.stack_activations(push_input)
        if dropout:
            push_input = F.dropout(push_input, p=dropout, training=self.training)
        if self.like_LSTM:
            data_gate = self.data_linear(inp)
            data_gate = torch.sigmoid(data_gate)
            push_input = data_gate * push_input
        push_indices = torch.tensor(range(0, self.stack_size-1), dtype=torch.long).to(inp.device)
        push = torch.index_select(stack, 1, push_indices)
        push = torch.cat([push_input, push], dim=1)

        # Pop
        pop_indices = torch.tensor(range(1, self.stack_size+1), dtype=torch.long).to(inp.device)
        pop = torch.index_select(stack, 1, pop_indices)

        # No op
        no_op_indices = torch.tensor(range(0, self.stack_size), dtype=torch.long).to(inp.device)
        no_op = torch.index_select(stack, 1, no_op_indices) # (batch_size, stack_size, d_model)

        # Calculate push, pop, and no_op gates
        pop_gate = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=inp.device)   
        no_op_gate = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=inp.device)               
        if self.gate_push_pop:
            tmp_push = self.push_gate_linear(inp)
            push_gate = torch.sigmoid(tmp_push)
            if not self.no_pop:
                tmp_pop = self.pop_gate_linear(inp)
                pop_gate = torch.sigmoid(tmp_pop)
            if self.no_op or self.no_pop:
                tmp_no_op = self.no_op_gate_linear(inp)
                no_op_gate = torch.sigmoid(tmp_no_op)
            if self.normalize_action:
                if self.no_op:
                    tmpAction = torch.cat([push_gate, pop_gate, no_op_gate], dim=1)
                    tmpAction = F.softmax(tmpAction, dim=1)
                    push_gate, pop_gate, no_op_gate = torch.split(tmpAction, 1, dim=1) 
                elif self.no_pop:
                    tmpAction = torch.cat([push_gate, no_op_gate], dim=1)
                    tmpAction = F.softmax(tmpAction, dim=1)
                    push_gate, no_op_gate = torch.split(tmpAction, 1, dim=1) 
                else:
                    tmpAction = torch.cat([push_gate, pop_gate], dim=1)
                    tmpAction = F.softmax(tmpAction, dim=1)
                    push_gate, pop_gate = torch.split(tmpAction, 1, dim=1)
        else:
            tmp = self.action(inp)
            action =  F.softmax(tmp, dim=2) # (batch_size, 1, 2) or (batch_size, 1, 3)
            if self.no_op:
                push_gate, pop_gate, no_op_gate = torch.split(action, 1, dim=2)
            elif self.no_pop:
                push_gate, no_op_gate = torch.split(action, 1, dim=2)
            else:
                push_gate, pop_gate = torch.split(action, 1, dim=2)

        return push_gate * push + pop_gate * pop + no_op_gate * no_op


class BinaryStack(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        assert not (params.no_op and params.no_pop) # at least one of these should be False

        self.d_model    = params.d_model
        self.stack_activations = torch.sigmoid if params.stack_activation == "sigmoid" else torch.tanh

        if params.gate_push_pop:
            self.push_gate_linear = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)
            if not params.no_pop:
                self.pop_gate_linear = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)
            if params.no_op or params.no_pop:
                self.no_op_gate_linear = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)
        else:
            if params.no_op:
                self.action = torch.nn.Linear(self.d_model * 2, 3, bias=True)
            else:
                self.action = torch.nn.Linear(self.d_model * 2, 2, bias=True)

        self.gate_linear_l = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)
        self.gate_linear_r = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)
        self.input_linear = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)
        if params.like_LSTM:
            self.data_linear = torch.nn.Linear(self.d_model * 2, self.d_model, bias=True)

        self.stack_size = params.stack_size
        self.no_op      = params.no_op
        self.no_pop = params.no_pop
        self.like_LSTM = params.like_LSTM
        self.gate_push_pop = params.gate_push_pop
        self.normalize_action = params.normalize_action

    def forward(self, input_left, input_right, stack_left, stack_right, dropout=None):
        """
        Args:
            input_left, input_right: (batch_size, d_model)
            stack_left, stack_right: (batch_size, stack_size, d_model)

        Returns:
            (batch_size, stack_size, d_model)
        """
        inp = torch.cat((input_left, input_right), dim=1)
        batch_size = inp.size()[0]
        inp = inp.unsqueeze(1)
        
        # Push zeros onto stack as placeholder
        left_gate = self.gate_linear_l(inp)
        left_gate = torch.sigmoid(left_gate) # (batch_size, 1, d_model)
        right_gate = self.gate_linear_r(inp)
        right_gate = torch.sigmoid(right_gate) # (batch_size, 1, d_model)
        stack = left_gate * stack_left + right_gate * stack_right
        zeros = torch.zeros(
                    (batch_size, 1, self.d_model), dtype=torch.float, requires_grad=True, device=inp.device
                )
        stack = torch.cat([stack, zeros], dim=1)

        # Push
        push_input = self.input_linear(inp) # (batch_size, 1, d_model)
        push_input = self.stack_activations(push_input)
        if dropout:
            push_input = F.dropout(push_input, p=dropout, training=self.training)
        if self.like_LSTM:
            data_gate = self.data_linear(inp)
            data_gate = torch.sigmoid(data_gate)
            push_input = data_gate * push_input
        push_indices = torch.tensor(range(0, self.stack_size-1), dtype=torch.long).to(inp.device)
        push = torch.index_select(stack, 1, push_indices)
        push = torch.cat([push_input, push], dim=1)

        # Pop
        pop_indices = torch.tensor(range(1, self.stack_size+1), dtype=torch.long).to(inp.device)
        pop = torch.index_select(stack, 1, pop_indices)

        # No op
        no_op_indices = torch.tensor(range(0, self.stack_size), dtype=torch.long).to(inp.device)
        no_op = torch.index_select(stack, 1, no_op_indices) # (batch_size, stack_size, d_model)

        # Calculate push, pop, and no_op gates
        pop_gate = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=inp.device)   
        no_op_gate = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=inp.device)            
        if self.gate_push_pop:
            tmp_push = self.push_gate_linear(inp)
            push_gate = torch.sigmoid(tmp_push)
            if not self.no_pop:
                tmp_pop = self.pop_gate_linear(inp)
                pop_gate = torch.sigmoid(tmp_pop)
            if self.no_op or self.no_pop:
                tmp_no_op = self.no_op_gate_linear(inp)
                no_op_gate = torch.sigmoid(tmp_no_op)
            if self.normalize_action:
                if self.no_op:
                    tmpAction = torch.cat([push_gate, pop_gate, no_op_gate], dim=1)
                    tmpAction = F.softmax(tmpAction, dim=1)
                    push_gate, pop_gate, no_op_gate = torch.split(tmpAction, 1, dim=1) 
                elif self.no_pop:
                    tmpAction = torch.cat([push_gate, no_op_gate], dim=1)
                    tmpAction = F.softmax(tmpAction, dim=1)
                    push_gate, no_op_gate = torch.split(tmpAction, 1, dim=1) 
                else:
                    tmpAction = torch.cat([push_gate, pop_gate], dim=1)
                    tmpAction = F.softmax(tmpAction, dim=1)
                    push_gate, pop_gate = torch.split(tmpAction, 1, dim=1)
        else:
            tmp = self.action(inp)
            action =  F.softmax(tmp, dim=2) # (batch_size, 1, 2) or (batch_size, 1, 3)
            if self.no_op:
                push_gate, pop_gate, no_op_gate = torch.split(action, 1, dim=2)
            elif self.no_pop:
                push_gate, no_op_gate = torch.split(action, 1, dim=2)
            else:
                push_gate, pop_gate = torch.split(action, 1, dim=2)

        return push_gate * push + pop_gate * pop + no_op_gate * no_op
