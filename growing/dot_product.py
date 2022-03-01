import torch

from .base import GrowingModule

from contextlib import contextmanager
from typing import Optional


class DotProduct(GrowingModule):
    def __init__(self, d_model, heads, d_head):
        super().__init__()

        self.q_linear = torch.nn.Linear(d_model, heads*d_head)
        self.k_linear = torch.nn.Linear(d_model, heads*d_head)

        self.heads = heads

        self.d_head = d_head

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

        return product  / torch.sqrt(torch.tensor(self.d_head))

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

            d_new = selected.size(0) / self.heads

            q_weight = torch.empty(
                self.heads, self.d_head + d_new, self.in_features
            )

            k_weight = torch.empty(
                self.heads, self.d_head + d_new, self.in_features
            )

            q_bias = torch.empty(self.heads, self.d_head + d_new)

            k_bias = torch.empty(self.heads, self.d_head + d_new)

            # copy old neurons

            q_weight[:, :self.d_head] = self.q_linear.weight
            k_weight[:, :self.d_head] = self.k_linear.weight
            q_bias[:, :self.d_head] = self.q_linear.bias
            k_bias[:, :self.d_head] = self.k_linear.bias

            # copy new neurons
            q_weight[:, self.d_head:] = self._weight_dir[0][selected] * sef.new_neurons[selected]
            k_weight[:, self.d_head:] = self._weight_dir[1][selected]
            q_bias[:, self.d_head:] = self._weight_bias[0][selected] * sef.new_neurons[selected]
            k_bias[:, self.d_head:] = self._weight_bisa[1][selected]

            scale = torch.sqrt(torch.tensor(new_d_head / self.d_head))

            q_weight /= scale
            k_weight /= scale
            q_bias /= scale
            k_bias /= scale

            self.q_linear.weight = torch.nn.Parameter(q_weight.reshape(self.heads, self.d_head, -1))
            self.k_linear.weight = torch.nn.Parameter(k_weight.reshape(self.heads, self.d_head, -1))
            self.q_linear.bias = torch.nn.Parameter(q_bias.reshape(self.heads, self.d_head, -1))
            self.k_linear.bias = torch.nn.Parameter(k_bias.reshape(self.heads, self.d_head, -1))


        self.reset_grow_state()
