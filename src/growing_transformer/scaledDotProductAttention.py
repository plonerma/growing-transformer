from typing import Any, Iterable, Mapping, Optional

import torch

from .base import GrowingModule


class ScaledDotProductAttention(GrowingModule):
    def __init__(self, d_model: int, heads: int, d_head: int, batch_first: bool = True, config: Mapping[str, Any] = {}):
        super().__init__(config)

        self.query_linear = torch.nn.Linear(d_model, heads * d_head)
        self.key_linear = torch.nn.Linear(d_model, heads * d_head)

        self.heads = heads

        self.d_head = d_head
        self.batch_first = batch_first
        self.reset_grow_state()

    def reset_grow_state(self) -> None:
        # step size (used to calculate gradients for selecting kept neurons)
        self.new_neurons = None

        # update directions (to be trained)
        self._weight: Optional[torch.nn.Parameter] = None
        self._bias: Optional[torch.nn.Parameter] = None

    def _direction_params(self) -> Iterable[Optional[torch.nn.Parameter]]:
        return [
            self._weight,
            self._bias,
        ]

    @property
    def d_model(self) -> int:
        return self.query_linear.weight.size(1)

    @property
    def in_features(self) -> int:
        return self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x.transpose(1, 0)

        length, batch_size, _ = x.size()

        q = self.query_linear(x).view(length, batch_size, self.heads, -1)

        k = self.key_linear(x).view(length, batch_size, self.heads, -1)

        # Batch (b) and head (h) dimensions remain intact
        # Query dimension (q) remains in the same place
        # Embedding values (e) are used in actual dot product
        einsum_str = "qbhe,kbhe->bhqk"
        product: torch.Tensor = torch.einsum(einsum_str, q, k)

        if self.new_neurons is not None:
            assert self._weight is not None and self._bias is not None

            q_novel = torch.nn.functional.linear(x, self._weight[0], self._bias[0])
            q_novel = q_novel.view(length, batch_size, self.heads, -1)
            q_novel = q_novel * self.new_neurons

            k_novel = torch.nn.functional.linear(x, self._weight[1], self._bias[1])
            k_novel = k_novel.view(length, batch_size, self.heads, -1)

            product = product + torch.einsum(einsum_str, q_novel, k_novel)

        product = product / torch.sqrt(torch.tensor(self.d_head))

        return torch.softmax(product, dim=-1)

    def grow(self) -> torch.Size:
        step_size = self.get_config("step_size", default=1e-2)
        num_novel = self.get_config("num_novel", default=0)
        eps_novel_weight = self.get_config("eps_novel_weight", "eps_novel", default=1e-1)
        eps_novel_bias = self.get_config("eps_novel_bias", "eps_novel", default=1e-1)

        # add parameter to measure influence/gradient of adding new neurons
        self.new_neurons = torch.nn.Parameter(torch.ones(self.heads, num_novel) * step_size, requires_grad=False)

        # create update direction for weight and bias
        self._weight = torch.nn.Parameter(torch.empty(2, self.heads * num_novel, self.d_model), requires_grad=False)

        self._bias = torch.nn.Parameter(torch.empty(2, self.heads * num_novel), requires_grad=False)

        torch.nn.init.uniform_(self._weight, -eps_novel_weight, eps_novel_weight)

        torch.nn.init.uniform_(self._bias, -eps_novel_bias, eps_novel_bias)

        return self.new_neurons.size()

    def degrow(self, selected: torch.Tensor) -> None:
        with torch.no_grad():

            if self.new_neurons is None:
                return

            assert self._weight is not None
            assert self._bias is not None

            d_new = selected.size(0) // self.heads

            q_weight = torch.empty(self.heads, self.d_head + d_new, self.in_features)

            k_weight = torch.empty(self.heads, self.d_head + d_new, self.in_features)

            q_bias = torch.empty(self.heads, self.d_head + d_new)

            k_bias = torch.empty(self.heads, self.d_head + d_new)

            # copy old neurons

            q_weight[:, : self.d_head] = self.query_linear.weight.view(self.heads, self.d_head, self.d_model)
            k_weight[:, : self.d_head] = self.key_linear.weight.view(self.heads, self.d_head, self.d_model)
            q_bias[:, : self.d_head] = self.query_linear.bias.view(self.heads, self.d_head)
            k_bias[:, : self.d_head] = self.key_linear.bias.view(self.heads, self.d_head)

            # copy new neurons
            selected_steps = self.new_neurons.view(-1)[selected].view(self.heads, -1)

            # temporarily merge heads and output dimension
            selected_weight = self._weight.view(2, -1, self.d_model)
            # only use selected neurons
            selected_weight = selected_weight[:, selected]
            # split up dimensions again
            selected_weight = selected_weight.view(2, self.heads, -1, self.d_model)

            # same for bias
            selected_bias = self._bias.view(2, -1)[:, selected].view(2, self.heads, -1)

            q_weight[:, self.d_head :] = selected_weight[0] * selected_steps[..., None]
            k_weight[:, self.d_head :] = selected_weight[1]
            q_bias[:, self.d_head :] = selected_bias[0] * selected_steps
            k_bias[:, self.d_head :] = selected_bias[1]

            scale = torch.sqrt(torch.tensor(1 + d_new / self.d_head))

            q_weight *= scale
            q_bias *= scale

            self.d_head = self.d_head + d_new

            self.query_linear.weight = torch.nn.Parameter(q_weight.reshape(self.heads * self.d_head, self.d_model))
            self.key_linear.weight = torch.nn.Parameter(k_weight.reshape(self.heads * self.d_head, self.d_model))
            self.query_linear.bias = torch.nn.Parameter(q_bias.reshape(self.heads * self.d_head))
            self.key_linear.bias = torch.nn.Parameter(k_bias.reshape(self.heads * self.d_head))

        self.reset_grow_state()
