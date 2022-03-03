import torch

from .base import GrowingModule
from .dot_product import DotProduct

from contextlib import contextmanager
from typing import Optional


class MultiheadAttention(GrowingModule):
    def __init__(self, d_model, heads, d_head):
        super().__init__()

        self.dot_product = DotProduct(d_model, heads, d_head)

        self.v_linear = torch.nn.Linear(d_model, heads*d_head)
        self.final_linear = torch.nn.Linear(heads*d_head, d_model)

        self.heads = heads
        self.d_head = d_head

    def reset_grow_state(self):
        super().reset_grow_state()

        # update directions (to be trained)
        self._weight_dir = None
        self._bias_dir = None

    def direction_params(self):
        return [
            self._weight_dir,
            self._bias_dir
        ]

    @property
    def in_features(self):
        return self.d_model

    def forward(self, x):
        batch_size, length, _ = x.size()

        q = self.q_linear(x).view(batch_size, length, self.heads, -1)
        k = self.k_linear(x).view(batch_size, length, self.heads, -1)

        # Batch (b) and head (h) dimensions remain intact
        # Query dimension (q) remains in the same place
        product = torch.einsum('bqhi,bkhi->bqhk', q, k)

        if self.new_neurons is not None:
            q_novel = torch.nn.functional.linear(x, self._weight_dir[0], self._bias_dir[0]).view(batch_size, length, self.heads, -1)

            q_novel = q_novel * self.new_neurons

            k_novel = torch.nn.functional.linear(x, self._weight_dir[1], self._bias_dir[1]).view(batch_size, length, self.heads, -1)

            product = product + torch.einsum('bqhi,bkhi->bqhk', q_novel, k_novel)

        return torch.softmax(
            product  / torch.sqrt(torch.tensor(self.d_head)),

             # The attention scores (given a batch, head, & query) should sum up to 1
            axis=-1
        )

    def grow(self,
             num_novel : int = 0,
             step_size = 1,
             eps_novel : float = 1e-2,
             eps_novel_weight : Optional[float] = None,
             eps_novel_bias : Optional[float] = None,
             **kw):
        # add parameter to measure influence/gradient of adding new neurons
        self.new_neurons = torch.nn.Parameter(
            torch.ones(self.heads * num_novel) * step_size,
            requires_grad=False
        )

        # create update direction for weight and bias
        self._weight_dir = torch.nn.Parameter(torch.empty(
            2, self.heads * num_novel, self.in_features
        ), requires_grad=False)

        self._bias_dir = torch.nn.Parameter(torch.empty(
            2, self.heads * num_novel
        ), requires_grad=False)

        e = eps_novel_weight or eps_novel
        torch.nn.init.uniform_(
            self._weight_dir,
            -e, e)

        e = eps_novel_bias or eps_novel
        torch.nn.init.uniform_(
            self._bias_dir,
            -e, e)

    def degrow(self, selected : torch.Tensor):
        with torch.no_grad():

            d_new = selected.size(0) / self.heads

            q_weight = torch.empty(
                self.heads, self.d_head + d_new, self.in_features
            )

            k_weight = torch.empty(
                self.heads, self.d_head + d_new, self.in_features
            )

            q_bias = torch.empty(self.heads, self.d_head + d_new)

            k_bias = torch.empty(self.heads, self.d_head + d_new)

            scale = torch.sqrt(torch.tensor(new_d_head / self.d_head))

            # copy old neurons

            q_weight[:, :self.d_head] = self.q_linear.weight * scale
            k_weight[:, :self.d_head] = self.k_linear.weight * scale
            q_bias[:, :self.d_head] = self.q_linear.bias * scale
            k_bias[:, :self.d_head] = self.k_linear.bias * scale

            # copy new neurons
            selected_steps = self.new_neurons.view(-1)[selected].view(self.heads, -1)
            selected_weight = self._weight_dir.view(2, self.heads * self.d_head, -1)[selected].view(2, self.heads, self.d_head, -1)
            selected_bias = self._bias_dir.view(2, self.heads * self.d_head)[selected].view(2, self.heads, self.d_head)

            q_weight[:, self.d_head:] = selected_weight[0] * selected_steps
            k_weight[:, self.d_head:] = selected_weight[1]
            q_bias[:, self.d_head:] = selected_bias[0] * selected_steps
            k_bias[:, self.d_head:] = selected_bias[1]

            self.q_linear.weight = torch.nn.Parameter(q_weight.reshape(self.heads, self.d_head, -1))
            self.k_linear.weight = torch.nn.Parameter(k_weight.reshape(self.heads, self.d_head, -1))
            self.q_linear.bias = torch.nn.Parameter(q_bias.reshape(self.heads, self.d_head, -1))
            self.k_linear.bias = torch.nn.Parameter(k_bias.reshape(self.heads, self.d_head, -1))


        self.reset_grow_state()
