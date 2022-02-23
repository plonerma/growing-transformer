import torch
from torch import Tensor

import math
from .util import slice1d

class Linear(torch.nn.Linear):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.reset_grow_state()

    def reset_grow_state(self):
        self.grown_weight = None
        self.grown_bias = None
        self.v_weight = None
        self.v_bias = None

    def forward(self, input: Tensor) -> Tensor:
        if self.grown_weight is not None:
            return F.linear(input, self.grown_weight, self.grown_bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def grow(self,
             dim : int = 0,  # dimension that is grown
             split : bool = True,  # whether to apply neuron splitting
             divide : bool = False, # wehther to divide split neurons
             num_novel : int = 0,  # number of totally new neurons
             eps_novel : float = 1.0,
             eps_split : float = 1e-6
             ):

        # calculate new size of weight tensor
        size = list(self.weight.size())

        if split:
            size[dim] *= 2

        size[dim] += num_novel

        # initialize new weight vector
        weight = torch.empty(*size)

        # copy original weights + small noise
        noise = torch.empty_like(self.weight)
        torch.nn.init.normal_(noise, 0, eps_split)

        weight[:self.weight.size(0), :self.weight.size(1)] = self.weight.data + noise

        # copy split weights - small noise
        offset = [0,0]
        offset[dim] = self.weight.size(dim)

        extent = list(self.weight.size())
        extent[dim] *= 2

        weight[offset[0]:extent[0], offset[1]:extent[1]] = self.weight.data - noise

        if divide:
            weight *= 0.5

        # randomly initialize other weights
        stdv = eps_novel / math.sqrt(weight.size(1))

        offset[dim] = 2*self.weight.size(dim)

        torch.nn.init.uniform_(weight[offset[0]:, offset[1]:], -stdv, stdv)

        self.weight = torch.nn.Parameter(weight)

        # only grow if there even is a bias and if we are growing the output size
        if self.bias is not None and dim == 0:
            # grow bias
            bias = torch.empty(size[0])

            noise = torch.empty_like(self.bias)
            torch.nn.init.normal_(noise, 0, eps_split)

            # copy original bias
            bias[:self.bias.size(0)] = self.bias.data + noise

            # copy split bias
            bias[self.bias.size(0):2*self.bias.size(0)] = self.bias.data - noise

            if divide:
                bias *= 0.5

            # initialize novel neurons
            torch.nn.init.uniform_(bias[2*self.bias.size(0):], -stdv, stdv)

            self.bias = torch.nn.Parameter(bias)

        # adjust features
        self.out_features, self.in_features = self.weight.size()

    def degrow(self,
               selected : torch.Tensor,  # indices of newly added neurons to keep
               dim :int = 0,  # dimension that was grown
               split : bool = True,  # whether neuron splitting was applied
               num_old : int = 0,  # number of old neurons
              ):
        # split neurons to keep
        split = selected[selected < num_old]

        # binary mask of split to keep
        kept_splits = torch.zeros(num_old)
        kept_splits[split] = True

        # calculate target size
        size = list(self.weight.size())
        size[dim] = num_old + selected.size(0)

        weight = torch.empty(size)

        slice_dim = slice1d(dim)

        # copy old neurons (unsplit neurons will be overwritten)
        weight[slice_dim[:num_old]] = self.prev_weight[slice_dim[:num_old]]

        # keep first copy of split neurons
        weight[slice_dim[split]] = self.weight.data[slice_dim[split]]

        # copy kept new neurons
        weight[slice_dim[num_old:]] = self.weight.data[slice_dim[num_old + selected]]

        self.weight = torch.nn.Parameter(weight)

        if dim == 0 and self.bias is not None:
            bias = torch.empty(num_old + len(selected))

            # copy old neurons (unsplit neurons will be overwritten)
            bias[:num_old] = self.prev_bias[:num_old]

            # keep first copy of split neurons
            bias[split] = self.bias.data[split]

            # copy kept new neurons
            bias[num_old:] = self.bias.data[num_old + selected]


            self.bias = torch.nn.Parameter(bias)

        # adjust features
        self.out_features, self.in_features = self.weight.size()
