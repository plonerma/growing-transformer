from typing import Optional

from transformers import BertConfig


def first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg
    return arg


class GrowingConfig(BertConfig):
    model_type = "GrowingBert"
    attribute_map = {"d_model": "hidden_size", "num_heads": "num_attention_heads"}

    def __init__(
        self,
        *args,
        num_hidden_layers: int = 3,
        d_head: int = 64,
        d_model: int = 768,
        num_heads: int = 12,
        step_size: float = 1e-1,
        bert_like_state_dict: bool = True,
        split: bool = True,
        mlp_split: Optional[bool] = None,
        num_novel: int = 4,
        mlp_num_novel: Optional[int] = None,
        mha_num_novel: Optional[int] = None,
        sdp_num_novel: Optional[int] = None,
        eps_split: float = 1e-1,
        eps_novel: float = 1e-1,
        eps_split_weight: Optional[float] = None,
        eps_split_bias: Optional[float] = None,
        eps_novel_weight: Optional[float] = None,
        eps_novel_bias: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.d_head = d_head
        self.d_model = d_model
        self.num_heads = num_heads
        self.step_size = step_size
        self.bert_like_state_dict = bert_like_state_dict

        self.mlp_split = first_not_none(mlp_split, split)

        # number of neurons that are introduced
        self.mlp_num_novel = first_not_none(mlp_num_novel, num_novel)
        self.mha_num_novel = first_not_none(mha_num_novel, num_novel)
        self.sdp_num_novel = first_not_none(sdp_num_novel, num_novel)

        # values for initializing split directions and novel neurons
        self.eps_split_weight = first_not_none(eps_split_weight, eps_split)
        self.eps_split_bias = first_not_none(eps_split_bias, eps_split)
        self.eps_novel_weight = first_not_none(eps_novel_weight, eps_novel)
        self.eps_novel_bias = first_not_none(eps_novel_bias, eps_novel)
