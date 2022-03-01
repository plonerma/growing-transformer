import torch

from .base import GrowingModule

from contextlib import contextmanager
from typing import Optional


class Attention(GrowingModule):
    def __init__(self, d_model, heads, d_head):
        super().__init__()

        self.q_linear = torch.nn.Linear(d_model, heads*d_head)
        self.k_linear = torch.nn.Linear(d_model, heads*d_head)

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
             eps_novel_bias : Optional[float] = None):
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

            num_old = self.in_features


            q_weight = torch.empty(
                num_old + selected.size(0),
                self.in_features
            )

            k_weight = torch.empty(
                num_old + selected.size(0),
                self.in_features,
            )

            q_bias = torch.empty(
                num_old + selected.size(0),
            )

            k_bias = torch.empty(
                num_old + selected.size(0),
            )

            num_new = selected.size(0) / self.heads
            new_d_head = self.d_head + num_new
            scale = torch.sqrt(torch.tensor(new_d_head / self.d_head))

            # copy old neurons
            # space out entries

            head_offsets = torch.arange(self.heads) * (self.d_head + num_new)
            old_indices = (
                head_offsets.repeat_interleave(d_head)
                # per head: enumerate old neurons
                + torch.arange(self.d_head).repeat(self.heads)
            )

            q_weight[old_indices] = self.q_linear.weight * scale
            k_weight[old_indices] = self.k_linear.weight * scale
            q_bias[old_indices] = self.q_linear.bias * scale
            k_bias[old_indices] = self.k_linear.bias * scale

            new_indices = (
                # offset per head
                head_offsets.repeat_interleave(num_new)
                + torch.arange(num_new).repeat(self.heads) + self.d_head
            )

            # copy new neurons
            q_weight[new_indices] = self._weight_dir[0][selected] * sef.new_neurons[selected]
            k_weight[new_indices] = self._weight_dir[1][selected]
            q_bias[new_indices] = self._weight_bias[0][selected] * sef.new_neurons[selected]
            k_bias[new_indices] = self._weight_bisa[1][selected]

            self.q_linear.weight = torch.nn.Parameter(q_weight)
            self.k_linear.weight = torch.nn.Parameter(k_weight)
            self.q_linear.bias = torch.nn.Parameter(q_bias)
            self.k_linear.bias = torch.nn.Parameter(k_bias)


        self.reset_grow_state()
