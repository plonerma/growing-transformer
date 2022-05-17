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
        num_hidden_layers: int = 12,
        layer_norm_eps: float = 1e-9,
        d_head: int = 64,
        d_head_kq: Optional[int] = None,
        d_head_v: Optional[int] = None,
        d_model: int = 768,
        num_heads: int = 12,
        step_size: float = 1e-1,
        layer_step_size: Optional[float] = None,
        bert_like_state_dict: bool = True,
        split: bool = True,
        mlp_split: Optional[bool] = None,
        num_novel: int = 4,
        mlp_num_novel: Optional[int] = None,
        mha_num_novel: Optional[int] = None,
        sdp_num_novel: Optional[int] = None,
        init_split_range: Optional[float] = None,
        loss_on_all_tokens=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # actual architecture
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers

        self.d_head = d_head
        self.d_head_kq = first_not_none(d_head_kq, d_head)
        self.d_head_v = first_not_none(d_head_v, d_head)

        self.d_model = d_model
        self.num_heads = num_heads
        self.bert_like_state_dict = bert_like_state_dict

        self.loss_on_all_tokens = loss_on_all_tokens

        # growth specific args
        self.step_size = step_size
        self.layer_step_size = first_not_none(layer_step_size, step_size)
        self.mlp_split = first_not_none(mlp_split, split)

        # number of neurons that are introduced
        self.mlp_num_novel = first_not_none(mlp_num_novel, num_novel)
        self.mha_num_novel = first_not_none(mha_num_novel, num_novel)
        self.sdp_num_novel = first_not_none(sdp_num_novel, num_novel)

        # values for initializing split directions and novel neurons
        self.init_split_range = first_not_none(init_split_range, self.initializer_range)
