import torch
from torch import Tensor
from torch.nn import functional as F

import math

class Linear(torch.nn.Linear):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.reset_grow_state()

    def reset_grow_state(self):
        # update directions (to be trained)
        self.weight_dir = None
        self.bias_dir = None

        # grown weights (resulting from update directions)
        self.grown_weight = None
        self.grown_bias = None

        self.was_split = False

    def forward(self, input: Tensor) -> Tensor:
        if self.grown_weight is not None:
            return F.linear(input, self.grown_weight, self.grown_bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def grow(self,
             dim : int = 0,  # dimension that is grown
             split : bool = True,  # whether to apply neuron splitting
             num_novel : int = 0,  # number of new neurons (excluding split)
             eps_novel : float = 1e-5,
             eps_split : float = 1e-5
             ):
        prev_h = self.weight.size(dim)

        self.was_split = split

        # size of update direction tensor
        size = list(self.weight.size())
        size[dim] = num_novel + split * prev_h

        # create update direction for weight
        self.weight_dir = torch.nn.Parameter(torch.empty(size), requires_grad=True)

        weight_dir = self.weight_dir

        if dim == 1:
            weight_dir = weight_dir.T

        # initialize split portion
        if split:
            torch.nn.init.normal_(weight_dir[:prev_h], 0, eps_split)

        # initialize novel portion
        if num_novel > 0:
            torch.nn.init.normal_(weight_dir[-num_novel:], 0, eps_novel)


        if dim == 0:
            self.bias_dir = torch.nn.Parameter(torch.empty(num_novel + split * prev_h), requires_grad=True)

            # initialize split portion
            if split:
                torch.nn.init.normal_(self.bias_dir[:prev_h], 0, eps_split)

            # initialize novel portion
            if num_novel > 0:
                torch.nn.init.normal_(self.bias_dir[-num_novel:], 0, eps_novel)

    def update_grown_weight(self):
        weight_dir = self.weight_dir
        bias_dir = self.bias_dir

        prev_weight = self.weight.detach()
        prev_bias = self.bias.detach()


        dim = (bias_dir is None) * 1
        prev_h = self.weight.size(dim)

        num_novel = weight_dir.size(dim) - prev_h * self.was_split

        # from here one we can work with weight_dir as if dim ==0
        if dim == 1:
            weight_dir = weight_dir.T
            prev_weight = prev_weight.T

        grown_weight = list()

        if self.was_split:
            if dim == 1:
                prev_weight = prev_weight * 0.5

            # copy split weight
            grown_weight = [
                prev_weight + weight_dir[:prev_h],
                prev_weight - weight_dir[:prev_h],
            ]

        if num_novel > 0:
            # copy novel neurons
            grown_weight.append(weight_dir[-num_novel:])

        grown_weight = torch.concat(grown_weight)

        if dim == 1:
            grown_weight = grown_weight.T

        self.grown_weight = grown_weight

        if bias_dir is not None:
            grown_bias = []

            if self.was_split:
                grown_bias = [
                    prev_bias + bias_dir[:prev_h],
                    prev_bias - bias_dir[:prev_h]
                ]

            if num_novel > 0:
                grown_bias.append(bias_dir[-num_novel:])

            self.grown_bias = torch.concat(grown_bias)

        # adjust features
        self.out_features, self.in_features = self.grown_weight.size()

    def degrow(self,
               selected : torch.Tensor,  # indices of newly added neurons to keep
               dim :int = 0,  # dimension that was grown
               split : bool = True,  # whether neuron splitting was applied
               num_old : int = 0,  # number of old neurons
              ):

        with torch.no_grad():
            # split neurons to keep
            split = selected[selected < num_old]

            # binary mask of split to keep
            kept_splits = torch.zeros(num_old)
            kept_splits[split] = True

            # calculate target size
            size = list(self.weight.size())
            size[dim] = num_old + selected.size(0)

            weight = torch.empty(size)
            prev_weight = self.weight.data
            grown_weight = self.grown_weight

            if dim == 1:
                weight = weight.T
                prev_weight = prev_weight.T
                grown_weight = grown_weight.T

            # copy old neurons (split neurons will be overwritten)
            weight[:num_old] = prev_weight

            # keep first copy of split neurons
            weight[split] = grown_weight[split]

            # copy second copy of split neurons + novel neurons
            weight[num_old:] = grown_weight[num_old + selected]

            if dim == 1:
                weight = weight.T

            self.weight = torch.nn.Parameter(weight)

            if dim == 0 and self.bias is not None:
                bias = torch.empty(num_old + len(selected))

                # copy old neurons (unsplit neurons will be overwritten)
                bias[:num_old] = self.bias[:num_old]

                # keep first copy of split neurons
                bias[split] = self.grown_bias[split]

                # copy kept new neurons
                bias[num_old:] = self.grown_bias[num_old + selected]

                self.bias = torch.nn.Parameter(bias)

            # adjust features
            self.out_features, self.in_features = self.weight.size()
            self.reset_grow_state()
