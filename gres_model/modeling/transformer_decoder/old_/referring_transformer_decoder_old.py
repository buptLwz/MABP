import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear
from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module.
"""
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B,-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedReferringDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        rla_weight: float = 0.1,
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        #self.lang_pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.att_weight = rla_weight
        self.self_attention_layers = nn.ModuleList()
        self.RIA_cross_attention = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        #self.lang_self_attention_layers = nn.ModuleList()
        #self.lang_cross_attention = nn.ModuleList()
        #self.lang_transformer_ffn_layers = nn.ModuleList()

        #self.lang_transformer_ffn_layers = nn.ModuleList()
        
        #self.lang_src_CA= nn.ModuleList()
        #self.lang_out_CA= nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.nt_embed = nn.ModuleList()
        self.mask_embed = nn.ModuleList()
        #self.lang_mask_embed = nn.ModuleList()
        #self.lang_decoder_norm = nn.ModuleList()
        #self.langsrc_transformer_ffn_layers = nn.ModuleList()
        #self.langout_transformer_ffn_layers = nn.ModuleList()

        #self.fusion_self_attention_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.1,
                    normalize_before=pre_norm,
                )
            )

            self.RIA_cross_attention.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.1,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.1,
                    normalize_before=pre_norm,
                )
            )
            # self.lang_self_attention_layers.append(
            #     SelfAttentionLayer(
            #         d_model=hidden_dim,
            #         nhead=nheads,
            #         dropout=0.1,
            #         normalize_before=pre_norm,
            #     )
            # )

            # self.lang_cross_attention.append(
            #     CrossAttentionLayer(
            #         d_model=hidden_dim,
            #         nhead=nheads,
            #         dropout=0.1,
            #         normalize_before=pre_norm,
            #     )
            # )

            # self.lang_transformer_ffn_layers.append(
            #     FFNLayer(
            #         d_model=hidden_dim,
            #         dim_feedforward=dim_feedforward,
            #         dropout=0.1,
            #         normalize_before=pre_norm,
            #     )
            # )

            # self.fusion_self_attention_layers.append(
            #     SelfAttentionLayer(
            #         d_model=hidden_dim,
            #         nhead=nheads,
            #         dropout=0.1,
            #         normalize_before=pre_norm,
            #     )
            # )

        self.num_feature_levels = 3
        # for _ in range(self.num_feature_levels+1):
        
        #     self.lang_src_CA.append(
        #         CrossAttentionLayer(
        #         d_model=hidden_dim,
        #         nhead=nheads,
        #         dropout=0.0,
        #         normalize_before=pre_norm))

        #     self.lang_out_CA.append(
        #         CrossAttentionLayer(
        #         d_model=hidden_dim,
        #         nhead=nheads,
        #         dropout=0.0,
        #         normalize_before=pre_norm))
            
        #     self.langsrc_transformer_ffn_layers.append(
        #         FFNLayer(
        #             d_model=hidden_dim,
        #             dim_feedforward=dim_feedforward,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )
        #     self.langout_transformer_ffn_layers.append(
        #         FFNLayer(
        #             d_model=hidden_dim,
        #             dim_feedforward=dim_feedforward,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )
            
        for _ in range(self.num_feature_levels+1):    
            self.nt_embed.append(MLP(hidden_dim, hidden_dim, 1, 2)) # nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            self.decoder_norm.append( nn.LayerNorm(hidden_dim))

            # self.lang_mask_embed.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            # self.lang_decoder_norm.append( nn.LayerNorm(hidden_dim))
        #self.decoder_norm_2 = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # learnable query features
        #self.lang_query_feat = nn.Embedding(num_queries, 768)
        # learnable query p.e.
        #self.lang_query_embed = nn.Embedding(num_queries, 768)
##################################
        self.langq_proj = Linear(hidden_dim, hidden_dim)
        self.langq_norm = nn.LayerNorm(hidden_dim)
        self.langk_proj = Linear(768, hidden_dim)
        self.langv_proj = Linear(768, hidden_dim)
        self.lango_proj = Linear(hidden_dim, hidden_dim)
        #self.lango_norm = nn.LayerNorm(hidden_dim)
        #self.lang_query_embed_proj = Linear(768, hidden_dim)






        # self.lang_out_CA = CrossAttentionLayer(d_model=hidden_dim,nhead=nheads,dropout=0.0,normalize_before=pre_norm)
        # self.lang_src_CA = CrossAttentionLayer(d_model=hidden_dim,nhead=nheads,dropout=0.0,normalize_before=pre_norm)

        # self.RLA_proj = Linear(hidden_dim, hidden_dim, False)
        # self.RLA_gate = nn.parameter.Parameter(data=torch.as_tensor(0.))
        # self.RLA_lang = CrossAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=nheads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        # self.RLA_region = SelfAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=nheads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )

        
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        #nn.init.zeros_(self.lang_proj.weight)
        #nn.init.zeros_(self.RLA_proj.weight)

        # output FFNs
        # if self.mask_classification:
        #     self.minimap_embed = nn.Linear(hidden_dim, num_classes + 1)
        
    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = 1
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, lang_feat, lang_mask, lang_pwam):
        #print(mask_features.shape)
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        #mask = mask.squeeze(2)
        src = []
        pos = []
        size_list = []
        lang_src =[]
        lang_pos=[]
        # for i in lang_pwam:
        #     size_list.append(i.shape)
        #assert 1==2   
        #del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape)
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            #lang_pos.append(self.lang_pe_layer(lang_pwam[i], None).flatten(2))
            #lang_src.append(lang_pwam[i].flatten(2) + self.level_embed.weight[i][None, :, None])
            #lang_pwam[i] = lang_pwam[i].flatten(2).permute(2,0,1)

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
            #lang_pos[-1] = lang_pos[-1].permute(2, 0, 1)
            #lang_src[-1] = lang_src[-1].permute(2, 0, 1)
        # print(size_list)[torch.Size([3, 256, 15, 15]), torch.Size([3, 256, 30, 30]), torch.Size([3, 256, 60, 60]), torch.Size([3, 256, 15, 15]), torch.Size([3, 256, 30, 30]), torch.Size([3, 256, 60, 60])]
        # assert 1==2

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        channels = output.shape[-1]
        # lang_query_embed = self.lang_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        
        # lang_query_embed = self.lang_query_embed_proj(lang_query_embed)
        # lang_output = self.lang_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        lang_output = None
        lang_feat = lang_feat.permute(0,2,1) #B,Nl,768
        lang_q = self.langq_proj(output.permute(1,0,2))
        lang_q = self.langq_norm(lang_q)
        lang_k = self.langk_proj(lang_feat)*lang_mask #B,Nl,256
        lang_v = self.langv_proj(lang_feat)*lang_mask#B,Nl,256
        #lang_k = lang_k.reshape(bs,-1,channels//)
        sim_map = torch.einsum("blc,bqc->blq",lang_k,lang_q)
        sim_map = (channels ** -.5) * sim_map
        sim_map = sim_map + (1e4*lang_mask-1e4)
        output = torch.einsum("blc,blq->bqc",lang_v,sim_map)
        output = self.lango_proj(output)
        #output = self.lango_norm(output)
        output = output.permute(1,0,2)


        #predictions_minimap = []
        predictions_mask = []
        predictions_nt_label=[]


        #lang_feat = self.lang_proj(lang_feat.permute(2,0,1))

        outputs_mask_gathered, attn_mask, nt_label, lang_attn_mask, outputs_mask, output_gathers = self.forward_prediction_heads(output,lang_output,mask_features,lang_feat,
                                                                                   lang_mask,src,lang_src,size_list[0],pos,-1)
        
        # output = output.permute(1,0,2).repeat(1,1,4).reshape(bs,-1,channels).permute(1,0,2)
        # query_embed = query_embed.permute(1,0,2).repeat(1,1,4).reshape(bs,-1,channels).permute(1,0,2)
        # lang_output = lang_output.permute(1,0,2).repeat(1,1,4).reshape(bs,-1,channels).permute(1,0,2)
        # lang_query_embed = lang_query_embed.permute(1,0,2).repeat(1,1,4).reshape(bs,-1,channels).permute(1,0,2)


        # outputs_class, outputs_mask, attn_mask, tgt_mask, nt_label = self.forward_prediction_heads(
        #     output, output, mask_features, attn_mask_target_size=size_list[0])
        #predictions_minimap.append(outputs_class)
        # predictions_mask.append(outputs_mask_gathered)
        # predictions_nt_label.append(nt_label)

        # ReLA is applied multiple times for perfromance
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            #lang_attn_mask[torch.where(lang_attn_mask.sum(-1) == lang_attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.RIA_cross_attention[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # lang_output = self.lang_cross_attention[i](
            #     lang_output, lang_src[level_index],
            #     memory_mask=lang_attn_mask,
            #     memory_key_padding_mask=None,
            #     pos=lang_pos[level_index],query_pos=lang_query_embed
            # )

            # lang_output = self.lang_self_attention_layers[i](
            #     lang_output, tgt_mask=None,
            #     tgt_key_padding_mask=None,
            #     query_pos=lang_query_embed
            # )

            # fusion_out = self.fusion_self_attention_layers[i](
            #     torch.cat([output,lang_output],dim=0),tgt_mask=None,
            #     tgt_key_padding_mask=None,
            #     query_pos=torch.cat([query_embed,lang_query_embed],dim=0)
            # )

            # output = fusion_out[:fusion_out.shape[0]//2,:,:]
            # lang_output = fusion_out[fusion_out.shape[0]//2:,:,:]

            # # Postprocessing
            # lang_output = self.lang_transformer_ffn_layers[i](lang_output)

            output = self.transformer_ffn_layers[i](output)


            # apply a full RLA after RIA
            # Note: 
            # Performance is the same with the paper, but the training is not very stable
            # sometimes it may fail to converge
            # Please see the main branch for fix

            # if i == self.num_layers - 1:
            #     RLA_lang_feat = self.lang_proj(lang_feat.permute(0,2,1))
            #     RLA_lang_feat = self.RLA_lang(output, RLA_lang_feat.permute(1,0,2))
            #     RLA_vision_feat = self.RLA_region(output)
            #     RLA_out = self.RLA_proj(RLA_lang_feat + RLA_vision_feat) * F.sigmoid(self.RLA_gate)
            #     RLA_out = output + self.att_weight * RLA_out
            # else:
            #     RLA_out = output
            #channels = output.shape[0]
            if level_index==0 or (level_index==1):
                output = output.permute(1,0,2).repeat(1,1,2).reshape(bs,-1,4*channels)
                output = output.repeat(1,1,2).reshape(bs,-1,channels).permute(1,0,2)

                query_embed = query_embed.permute(1,0,2).repeat(1,1,2).reshape(bs,-1,4*channels)
                query_embed = query_embed.repeat(1,1,2).reshape(bs,-1,channels).permute(1,0,2)

                # lang_output = lang_output.permute(1,0,2).repeat(1,1,2).reshape(bs,-1,4*channels).permute(1,0,2)
                # lang_output = lang_output.permute(1,0,2).repeat(1,1,2).reshape(bs,-1,channels).permute(1,0,2)

                # lang_query_embed = lang_query_embed.permute(1,0,2).repeat(1,1,2).reshape(bs,-1,4*channels).permute(1,0,2)
                # lang_query_embed = lang_query_embed.permute(1,0,2).repeat(1,1,2).reshape(bs,-1,channels).permute(1,0,2)

            if level_index==2 and i<self.num_layers-1:
                sqrtq = torch.sqrt(torch.tensor(output.shape[0])).long() #qnc
                output = output.permute(1,2,0).reshape(bs,channels,sqrtq,sqrtq)
                output = F.avg_pool2d(output,(4,4),4).flatten(2).permute(2,0,1)
                #output = output.permute(1,0,2).repeat(1,1,4).reshape(bs,-1,channels).permute(1,0,2)
                query_embed = query_embed.permute(1,2,0).reshape(bs,channels,sqrtq,sqrtq)
                query_embed = F.avg_pool2d(query_embed,(4,4),4).flatten(2).permute(2,0,1)

                # lang_output = lang_output.permute(1,2,0).reshape(bs,channels,sqrtq,sqrtq)
                # lang_output = F.avg_pool2d(lang_output,(4,4),4).flatten(2).permute(2,0,1)

                # lang_query_embed = lang_query_embed.permute(1,2,0).reshape(bs,channels,sqrtq,sqrtq)
                # lang_query_embed = F.avg_pool2d(lang_query_embed,(4,4),4).flatten(2).permute(2,0,1)

                # query_embed = F.avg_pool2d(query_embed.permute(1,2,0),4,4).permute(2,0,1)
                # lang_output = F.avg_pool2d(lang_output.permute(1,2,0),4,4).permute(2,0,1)
                # lang_query_embed = F.avg_pool2d(lang_query_embed.permute(1,2,0),4,4).permute(2,0,1)
            #print(output.shape)
            outputs_mask_gathered, attn_mask, nt_label, lang_attn_mask, outputs_mask, output_gathers = self.forward_prediction_heads(output,lang_output,mask_features,lang_feat,
                                                                                   lang_mask,src,lang_src,size_list[(i+1)%self.num_feature_levels],pos,i)

            #predictions_minimap.append(outputs_class)
            if i > self.num_layers - self.num_feature_levels-2:
                predictions_mask.append(outputs_mask_gathered)
                predictions_nt_label.append(nt_label)


        out = {
            #'pred_logits': predictions_minimap[-1],
            'pred_masks': predictions_mask,
            'all_masks': outputs_mask,
            'all_gathers': output_gathers,
            'nt_label': predictions_nt_label
        }
        return out
    
    def forward_prediction_heads(self, output,lang_output, mask_features, lang_feat, mask, src, lang_src, srcsize, pos, i):
        numq = output.shape[0]
        #print(numq)
        # if i < self.num_layers:
        #     window_size = (srcsize[2]//(torch.sqrt(torch.tensor(numq)).long())+1)
        # else:
        #     window_size = (mask_features.shape[2]//(torch.sqrt(torch.tensor(numq)).long())+1)

        i = i + 1

        if i < self.num_layers-self.num_feature_levels:
            level_index = i % self.num_feature_levels
        else:
            level_index = i % (self.num_layers-self.num_feature_levels)

        decoder_output = self.decoder_norm[level_index](output)
        decoder_output = decoder_output.transpose(0, 1) #100,bs,256 -> bs,100,256
        #outputs_class = self.class_embed(decoder_output) #bs,100,256 -> bs,100,K+1
        mask_embed = self.mask_embed[level_index](decoder_output) #bs,100,256 -> bs,100,256(maskdim)
        
        #lang_decoder_output = self.lang_decoder_norm[level_index](lang_output)
        #lang_decoder_output = lang_decoder_output.transpose(0, 1) #100,bs,256 -> bs,100,256
        #outputs_class = self.class_embed(decoder_output) #bs,100,256 -> bs,100,K+1
        #lang_mask_embed = self.lang_mask_embed[level_index](lang_decoder_output) #bs,100,256 -> bs,100,256(maskdim)
        padl = 2**(level_index-1)
        if i < self.num_layers:
            window_size = (srcsize[2]//(torch.sqrt(torch.tensor(numq)).long())+1)
            #fold_params = dict(kernel_size=(srcsize[2]//torch.sqrt(numq)+1), dilation=1, padding=(srcsize[2]//torch.sqrt(numq)+1)-srcsize[2]%torch.sqrt(numq), stride=...)
            #lang_src = self.lang_src_CA[level_index](src[level_index],lang_feat,query_pos=pos[level_index],memory_key_padding_mask=mask)
            src_sw = src[level_index].permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #sbc->bchw
            #lang_src_sw = lang_src[level_index].permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #sbc->bchw
            
            #######################################
            if level_index == 0 :
                src_sw = F.pad(src_sw,(0,1,0,1))
                #lang_src_sw = F.pad(lang_src_sw,(0,1,0,1))
                
            else:
                
                src_sw = F.pad(src_sw,(padl,padl,padl,padl))
                #lang_src_sw = F.pad(lang_src_sw,(padl,padl,padl,padl))

            _,_,paddedH,paddedW = src_sw.shape
            #src[level_index] = F.unfold(src[level_index],**fold_params)
            # print(src_sw.shape)
            # print(window_size)
            src_sw = window_partition(src_sw.permute(0,2,3,1),window_size) #bchw->b,numq,wh,ww,c
            #print(src_sw.shape)
            outputs_mask_curlvl = torch.einsum("bqac,bqhwc->bqhwa",mask_embed.unsqueeze(2),src_sw)
            #print(outputs_mask_curlvl.shape)
            #print(outputs_mask_curlvl.flatten(0,1).shape)
            #print(outputs_mask_curlvl.flatten(0,1).shape)
            outputs_mask_curlvl = window_reverse(outputs_mask_curlvl.flatten(0,1),window_size,paddedH,paddedW) # b,q,wh,ww,a -> b,H,W,a


            #lang_src_sw = window_partition(lang_src_sw.permute(0,2,3,1),window_size)
            #lang_outputs_mask_curlvl = torch.einsum("bqac,bqhwc->bqhwa",lang_mask_embed.unsqueeze(2),lang_src_sw)
            #lang_outputs_mask_curlvl = window_reverse(lang_outputs_mask_curlvl.flatten(0,1),window_size,paddedH,paddedW) # b,q,wh,ww,a -> b,H,W,a
            #lang_outputs_mask_curlvl = lang_outputs_mask_curlvl[:,int(2**(i-1)-1):-int(2**(i-1)),int(2**(i-1)-1):-int(2**(i-1)),:] # b,q,wh,ww,a -> b,H,W,a   
            if level_index == 0 :
                outputs_mask_curlvl = outputs_mask_curlvl[:,:-1,:-1,:] # b,q,wh,ww,a -> b,H,W,a
                #lang_outputs_mask_curlvl = lang_outputs_mask_curlvl[:,:-1,:-1,:]
            else:
                outputs_mask_curlvl = outputs_mask_curlvl[:,int(padl):-int(padl),int(padl):-int(padl),:] # b,q,wh,ww,a -> b,H,W,a
                #lang_outputs_mask_curlvl = lang_outputs_mask_curlvl[:,int(padl):-int(padl),int(padl):-int(padl),:]
            #outputs_mask_curlvl = torch.einsum("bqc,sbc->sbq", mask_embed, src[level_index]).permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #bs,100,256 * HW,bs,256 -> bs,100,h,w
            #lang_outputs_mask_curlvl = torch.einsum("bqc,sbc->sbq", lang_mask_embed, lang_src[level_index]).permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #bs,100,256 * HW,bs,256 -> bs,100,h,w
        else:
            window_size = (mask_features.shape[2]//(torch.sqrt(torch.tensor(numq)).long())+1)
            #lang_src = self.lang_src_CA[level_index](mask_features.flatten(2).permute(2,0,1),lang_feat,memory_key_padding_mask=mask)
            mask_sw = F.pad(mask_features,(padl,padl,padl,padl))
            _,_,paddedH,paddedW = mask_sw.shape
            mask_sw = window_partition(mask_sw.permute(0,2,3,1),window_size)
            

            outputs_mask_curlvl = torch.einsum("bqac,bqhwc->bqhwa",mask_embed.unsqueeze(2),mask_sw)
            #lang_outputs_mask_curlvl = torch.einsum("bqac,bqhwc->bqhwa",lang_mask_embed.unsqueeze(2),mask_sw)
            outputs_mask_curlvl = window_reverse(outputs_mask_curlvl.flatten(0,1),window_size,paddedH,paddedW) # b,q,wh,ww,a -> b,H,W,a
            outputs_mask_curlvl = outputs_mask_curlvl[:,int(padl):-int(padl),int(padl):-int(padl),:] # b,q,wh,ww,a -> b,H,W,a

            # b,q,wh,ww,a -> b,H,W,a
            #lang_outputs_mask_curlvl = window_reverse(lang_outputs_mask_curlvl.flatten(0,1),window_size,paddedH,paddedW) # b,q,wh,ww,a -> b,H,W,a
            #lang_outputs_mask_curlvl = lang_outputs_mask_curlvl[:,int(padl):-int(padl),int(padl):-int(padl),:] # b,q,wh,ww,a -> b,H,W,a      
            # outputs_mask_curlvl = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) #bs,100,256 * bs,256,h,w -> bs,100,h,w
            # lang_outputs_mask_curlvl = torch.einsum("bqc,bchw->bqhw", lang_mask_embed, mask_features) #bs,100,256 * HW,bs,256 -> bs,100,h,w
        #outputs_mask_curlvl = torch.einsum("bqc,sbc->sbq", mask_embed, src[level_index]).permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #bs,100,256 * HW,bs,256 -> bs,100,h,w
        # if level_index==1 or i==self.num_layers-1:
        #     attn_mask = outputs_mask_curlvl.permute(0,3,1,2).repeat(1,numq*4,1,1) #b,H,W,a -> b,q,H,W
            
        #     lang_attn_mask = lang_outputs_mask_curlvl.permute(0,3,1,2).repeat(1,numq*4,1,1) #b,H,W,a -> b,q,H,W
        # else:
        attn_mask = outputs_mask_curlvl.permute(0,3,1,2).repeat(1,numq,1,1) #b,H,W,a -> b,q,H,W

        #lang_attn_mask = lang_outputs_mask_curlvl.permute(0,3,1,2).repeat(1,numq,1,1) #b,H,W,a -> b,q,H,W 
        #lang_attn_mask = -torch.mul(attn_mask,lang_attn_mask)
            
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool() #小于0.5的是True，会被mask掉，认为小于0.5的是背景，大于0.5的是前景
        attn_mask = attn_mask.detach()

       
        #lang_attn_mask = (lang_attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool() #小于0.5的是True，会被mask掉，认为小于0.5的是背景，大于0.5的是前景
        #lang_attn_mask = lang_attn_mask.detach()
        if i < self.num_layers-self.num_feature_levels:    
            return None, attn_mask, None, None, None, None
        else:
            nt_label = self.nt_embed[level_index](decoder_output)
            #nt_label = F.cosine_similarity(lang_decoder_output.unsqueeze(2),decoder_output.unsqueeze(1),dim=3)
            nt_label = nt_label.mean(dim=1)
            #gather = torch.sigmoid(torch.mul(lang_outputs_mask_curlvl,outputs_mask_curlvl))
            outputs_mask_gathered = outputs_mask_curlvl.squeeze(-1)#torch.sum((outputs_mask_curlvl,gather),dim=1)
            #nt_label = outputs_mask_gathered.mean(dim=(1,2))

            return outputs_mask_gathered, attn_mask, nt_label, None, outputs_mask_curlvl, None
        #     i = i + 1
        #     level_index = i % (self.num_layers-self.num_feature_levels)
        #     decoder_output = self.decoder_norm[level_index](output)
        #     decoder_output = decoder_output.transpose(0, 1) #100,bs,256 -> bs,100,256
        #     #outputs_class = self.class_embed(decoder_output) #bs,100,256 -> bs,100,K+1
        #     mask_embed = self.mask_embed[level_index](decoder_output) #bs,100,256 -> bs,100,256(maskdim)
        # # print(mask_embed.shape)
        # # print(lang_feat.shape)
        # # assert 1==2
        #     lang_query = self.lang_out_CA[level_index](mask_embed.permute(1,0,2),lang_feat,memory_key_padding_mask=mask)
        #     if i < self.num_layers:
        #         lang_src = self.lang_src_CA[level_index](src[level_index],lang_feat,query_pos=pos[level_index],memory_key_padding_mask=mask)
        #         outputs_mask_curlvl = torch.einsum("bqc,sbc->sbq", mask_embed, src[level_index]).permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #bs,100,256 * HW,bs,256 -> bs,100,h,w
        #     else:
        #         lang_src = self.lang_src_CA[level_index](mask_features.flatten(2).permute(2,0,1),lang_feat,memory_key_padding_mask=mask)
        #         outputs_mask_curlvl = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) #bs,100,256 * bs,256,h,w -> bs,100,h,w
            
        #     lang_src = self.langsrc_transformer_ffn_layers[level_index](lang_src)
        #     lang_query = self.langout_transformer_ffn_layers[level_index](lang_query)

        #     gather = torch.einsum('sbc,qbc->sbq',lang_src,lang_query).permute(1,2,0).reshape(outputs_mask_curlvl.shape)
        #     outputs_mask_gathered = torch.sum((outputs_mask_curlvl+gather),dim=1)
        #     # NOTE: prediction is of higher-resolution 哦prediction是指的mask_features，所以下边这个是个下采样，我原本一直以为是上采样，这里为啥非得下采样呢，就直接每层用每层的mask_features也没问题啊
        #     # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        #     #attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        #     # must use bool type
        #     # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.

        #     attn_mask = (outputs_mask_curlvl.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool() #小于0.5的是True，会被mask掉，认为小于0.5的是背景，大于0.5的是前景
        #     attn_mask = attn_mask.detach()
        #     nt_label = self.nt_embed[level_index](lang_query.permute(1,0,2))
        #     nt_label = nt_label.mean(dim=1)

            #return outputs_mask_gathered, attn_mask, nt_label, lang_attn_mask
        
        # else:

        #     level_index = i % self.num_feature_levels
        #     decoder_output = self.decoder_norm(output)
        #     decoder_output = decoder_output.transpose(0, 1) #100,bs,256 -> bs,100,256

        #     #outputs_class = self.class_embed(decoder_output) #bs,100,256 -> bs,100,K+1
        #     mask_embed = self.mask_embed(decoder_output) #bs,100,256 -> bs,100,256(maskdim)
        #     #lang_query = self.lang_out_CA[i](mask_embed,lang_feat.permute(1,0,2),query_pos=query_embed,pos=mask)
        #     #lang_src = self.lang_src_CA[i](src[level_index],lang_feat.permute(1,0,2),query_pos=pos[level_index],pos=mask)



        #     outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) #bs,100,256 * bs,256,h,w -> bs,100,h,w
        #     #gather = F.softmax(torch.einsum('sbc,qbc->sbq',lang_src,lang_query).permute(1,2,0).reshape(outputs_mask.shape),dim=1) 
        #     #outputs_mask = torch.mul(outputs_mask,gather)
        #     # NOTE: prediction is of higher-resolution 哦prediction是指的mask_features，所以下边这个是个下采样，我原本一直以为是上采样，这里为啥非得下采样呢，就直接每层用每层的mask_features也没问题啊
        #     # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        #     attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        #     # must use bool type
        #     # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        #     attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool() #小于0.5的是True，会被mask掉，认为小于0.5的是背景，大于0.5的是前景
        #     attn_mask = attn_mask.detach()

        #     return outputs_mask, attn_mask, lang_query

        # RLA feat -> minimap
        # RLA_feature = self.decoder_norm_2(RLA_feature)
        # RLA_feature = RLA_feature.transpose(0, 1)
        # outputs_minimap = self.minimap_embed(RLA_feature)

        # # _region feat -> mask
        # RIA_feature = self.decoder_norm(region_feature)
        # RIA_feature = RIA_feature.transpose(0, 1)
        # mask_embed = self.mask_embed(RIA_feature)
        
        # # RIA feat -> all masks
        # # q: region num
        # all_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        # # _region + minimap -> target mask
        # tgt_embed = torch.einsum("bqa,bqc->bac", outputs_minimap, mask_embed)
        # tgt_mask = torch.einsum("bac,bchw->bahw", tgt_embed, mask_features)

        # # RLA feat -> nt_label
        # nt_label = self.nt_embed(RLA_feature)
        # nt_label = nt_label.mean(dim=1)

        # attn_mask = F.interpolate(all_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        # attn_mask = attn_mask.detach()

        # return outputs_minimap, all_mask, attn_mask, tgt_mask, nt_label
    
    # def forward_prediction_heads(self, region_feature, RLA_feature, mask_features, attn_mask_target_size):
        
    #     # RLA feat -> minimap
    #     RLA_feature = self.decoder_norm_2(RLA_feature)
    #     RLA_feature = RLA_feature.transpose(0, 1)
    #     outputs_minimap = self.minimap_embed(RLA_feature)

    #     # _region feat -> mask
    #     RIA_feature = self.decoder_norm(region_feature)
    #     RIA_feature = RIA_feature.transpose(0, 1)
    #     mask_embed = self.mask_embed(RIA_feature)
        
    #     # RIA feat -> all masks
    #     # q: region num
    #     all_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
    #     # _region + minimap -> target mask
    #     tgt_embed = torch.einsum("bqa,bqc->bac", outputs_minimap, mask_embed)
    #     tgt_mask = torch.einsum("bac,bchw->bahw", tgt_embed, mask_features)

    #     # RLA feat -> nt_label
    #     nt_label = self.nt_embed(RLA_feature)
    #     nt_label = nt_label.mean(dim=1)

    #     attn_mask = F.interpolate(all_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
    #     attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
    #     attn_mask = attn_mask.detach()

    #     return outputs_minimap, all_mask, attn_mask, tgt_mask, nt_label
