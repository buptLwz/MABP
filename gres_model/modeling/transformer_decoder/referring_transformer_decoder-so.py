'''code for single-object MABP'''
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
import numpy as np

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, torch.div(H, window_size, rounding_mode='floor'), window_size, torch.div(W, window_size, rounding_mode='floor'), window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B,-1, window_size, window_size, C)
    window_id = torch.arange(((torch.div(H, window_size, rounding_mode='floor')) * (torch.div(W, window_size, rounding_mode='floor'))),device=windows.device).float().view(1,1,torch.div(H, window_size, rounding_mode='floor'), torch.div(W, window_size, rounding_mode='floor'))
    window_id = F.interpolate(window_id,(H,W),mode='nearest').repeat(B,1,1,1) #B,1,H,W
    return windows,window_id


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
    x = windows.view(B, torch.div(H, window_size, rounding_mode='floor'), torch.div(W, window_size, rounding_mode='floor'), window_size, window_size, -1)
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

'''MABP Decoder'''
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

        self.text_len = 20
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.num_heads = nheads
        self.num_layers = dec_layers
        # init for MMD
        self.MMD_self_attention = nn.ModuleList()
        self.MMD_cross_attention = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.layer_embed = nn.ModuleList()

        
        for _ in range(self.num_layers):
            self.MMD_self_attention.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0,
                    normalize_before=pre_norm,
                )
            )

            self.MMD_cross_attention.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0,
                    normalize_before=pre_norm,
                )
            )

            self.layer_embed.append(
                Linear(self.text_len,self.text_len)
            )

        # init for RSH
        self.num_feature_levels = 3
    
        self.lateral_convs=nn.ModuleList()
        self.output_convs=nn.ModuleList()

        #self.pooler_src_proj = nn.ModuleList()
        #self.nt_emb_proj = nn.ModuleList()
        #self.nt_lang_emb_proj = nn.ModuleList()

        self.num_queries = num_queries

        self.decoder_norm = nn.ModuleList()
        # self.nt_embed = nn.ModuleList()
        self.mask_embed = nn.ModuleList()

        for i in range(self.num_feature_levels+1):    

            #qq = (self.num_queries*(4**(min(i,self.num_feature_levels-1))))
            self.decoder_norm.append( nn.LayerNorm(hidden_dim))

            # main branch
            self.mask_embed.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            
            # no-target branch
            # self.pooler_src_proj.append(Linear(hidden_dim,hidden_dim))
            # self.nt_emb_proj.append(Linear(hidden_dim,hidden_dim))
            # self.nt_embed.append(MLP(2*qq, 2*qq, 2, 2))

            # self.nt_lang_emb_proj.append(MLP(self.text_len,self.text_len,1,2))
            

        # init for FPN
        for _ in range(self.num_feature_levels):    
            self.lateral_convs.append(Conv2d(
                hidden_dim, hidden_dim, kernel_size=1, bias=None,norm=None
            ))
            self.output_convs.append(Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=None,
                norm=None,
                activation=F.relu
            ))
            weight_init.c2_xavier_fill(self.lateral_convs[-1])
            weight_init.c2_xavier_fill(self.output_convs[-1])


        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # learnable language query p.e.
        self.lang_query_embed = nn.Embedding(self.text_len, hidden_dim)

        
        # init for query generator
        self.langq_proj = Linear(hidden_dim, hidden_dim)
        self.langq_norm = nn.LayerNorm(hidden_dim)
        self.langk_proj = Linear(768, hidden_dim)
        self.langv_proj = Linear(768, hidden_dim)
        self.lango_proj = Linear(hidden_dim, hidden_dim)

        # projector from extractor to decoder
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        
        #self.nt_weight = torch.tensor([0.01,0.1,0.5,1]).unsqueeze(0).unsqueeze(2)

        
    # Detectron2: init parameters directly from config
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

    def forward(self, f0,f1,f2, mask_features, lang_feat, lang_mask):

        src = []
        pos = []
        size_list = []

        '''project features from extractor'''
        x=[f0,f1,f2]

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape)
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)


        _, bs, _ = src[0].shape

        # QxNxC init query
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        lang_query_embed = self.lang_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        channels = output.shape[-1]

        # linguistic features from bert 
        lang_feat = lang_feat.permute(0,2,1) #B,Nl,768

        '''query generator'''
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
        lang_v = lang_v.permute(1,0,2)


        # for output
        predictions_mask = []
        #predictions_nt_label=[]
        output_fpn=[]


        for i in range(self.num_layers):
            level_index = torch.div(i, self.num_feature_levels, rounding_mode='floor') 
            
            # Control the interactive range of the query and set the all-zero mask to all-one to prevent numerical overflow.
            if i > 0:
                #attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False ##bhead,q,HW
                cur_attn_mask = torch.cat([attn_mask,torch.zeros((attn_mask.shape[0],attn_mask.shape[1],20),device=output.device,dtype=bool)],dim=-1)
            else:
                attn_mask = None
                cur_attn_mask = None


            lang_v = self.layer_embed[i](lang_v.permute(1,2,0)).permute(2,0,1)
            cur_src_lang = torch.cat([src[level_index],lang_v],dim=0) #HW,N,C -> HW+l,N,C
            cur_pos_lang = torch.cat([pos[level_index],lang_query_embed],dim=0) #HW,N,C -> HW+l,N,C

            '''MMD'''
           # attention: cross-attention first
            output = self.MMD_cross_attention[i](
                output, cur_src_lang,
                memory_mask=cur_attn_mask,
                memory_key_padding_mask=None,
                pos=cur_pos_lang, query_pos=query_embed
            )

            fusion_output = self.MMD_self_attention[i](
                torch.cat([output,lang_v],dim=0), tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=torch.cat([query_embed,lang_query_embed],dim=0)
            )

            fusion_output = self.transformer_ffn_layers[i](fusion_output)

            output = fusion_output[:output.shape[0],:,:]

            lang_v = fusion_output[output.shape[0]:,:,:]
            
            '''RSH'''

            '''Compute intermediate results only at the layer where the number of query is quadrupled and change visual features; 
            for the remaining layers, only perform MMD and forward (detach) the main branch to obtain the attention mask.'''

            if (i + 1) % self.num_feature_levels == 0:
                nt_lang = lang_v.permute(1,2,0) #b,256,20

                head_level_index = int(torch.div(i, self.num_feature_levels, rounding_mode='floor'))
                outputs_mask_gathered = \
                self.forward_prediction_heads(output, mask_features, src, size_list, head_level_index, False)#, nt_lang)
                predictions_mask.append(outputs_mask_gathered)
                #predictions_nt_label.append(nt_label)
                output_fpn.append(output)

                
                
                if level_index==0 or (level_index==1):
                    sqrtq = torch.sqrt(torch.tensor(output.shape[0],device=output.device)).long() #qnc
                    output = output.permute(1,2,0).reshape(bs,channels,sqrtq,sqrtq)
                    query_embed = query_embed.permute(1,2,0).reshape(bs,channels,sqrtq,sqrtq)
                    
                    output = F.interpolate(output, size=(sqrtq*2,sqrtq*2), mode="nearest")
                    query_embed = F.interpolate(query_embed, size=(sqrtq*2,sqrtq*2), mode="nearest")
                    output = output.flatten(2).permute(2,0,1)
                    query_embed = query_embed.flatten(2).permute(2,0,1)


                    amg_level_index = int(torch.div((i+1), self.num_feature_levels, rounding_mode='floor'))
                    attn_mask = self.attn_mask_generate(output, src, size_list, amg_level_index)

                if level_index==2:
                    '''FPN'''
                    for idx, x in enumerate(output_fpn):

                        sqrtq = torch.sqrt(torch.tensor(x.shape[0],device=output.device)).long()
                        x = x.permute(1,2,0).reshape(bs,channels,sqrtq,sqrtq)
                        if idx==0:
                            out = [x]
                        lateral_conv = self.lateral_convs[idx]
                        output_conv = self.output_convs[idx]
                        cur_fpn = lateral_conv(x)
                        # Following FPN implementation, we use nearest upsampling here
                        y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
                        y = output_conv(y)

                        out.append(y)

                    output_final = out[-1].flatten(2).permute(2,0,1)
                
                    head_level_index = int(torch.div(i+1, self.num_feature_levels, rounding_mode='floor'))
                    outputs_mask_gathered= \
                        self.forward_prediction_heads(output_final, mask_features, src, size_list, head_level_index, True)#,nt_lang)
                    predictions_mask.append(outputs_mask_gathered)
                    #predictions_nt_label.append(nt_label)

                
            else:
                amg_level_index = int(torch.div(i, self.num_feature_levels, rounding_mode='floor'))
                attn_mask = self.attn_mask_generate(output, src, size_list, amg_level_index)

        out = {
            'pred_masks': predictions_mask, 
            #'nt_label': predictions_nt_label # 4*B*2
        }
        return out
    
    def attn_mask_generate(self, output, src, srcsize, level_index):
        numq = output.shape[0]
        
        srcsize = srcsize[level_index]

        decoder_output = self.decoder_norm[level_index](output)
        decoder_output = decoder_output.transpose(0, 1) #100,bs,256 -> bs,100,256

        mask_embed = self.mask_embed[level_index](decoder_output) #bs,100,256 -> bs,100,256(maskdim)
        
        if level_index==0:
            padlt = 0
            padrb = 1

        else:
            padlt = 2**(level_index-1)
            padrb = 2**(level_index-1)


        window_size = torch.div(srcsize[2], (torch.sqrt(torch.tensor(numq,device=output.device)).long()), rounding_mode='floor')+1
        src_sw = src[level_index].permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #sbc->bchw

        src_sw = F.pad(src_sw,(int(padlt),int(padrb),int(padlt),int(padrb)))


        _,_,paddedH,paddedW = src_sw.shape

        src_sw,window_ids = window_partition(src_sw.permute(0,2,3,1),window_size) #bchw->b,numq,wh,ww,c B1HW

        outputs_mask_curlvl = torch.einsum("bqac,bqhwc->bqhwa",mask_embed.unsqueeze(2),src_sw)

        outputs_mask_curlvl = window_reverse(outputs_mask_curlvl.flatten(0,1),window_size,paddedH,paddedW) # b,q,wh,ww,a -> b,H,W,a

        outputs_mask_curlvl = outputs_mask_curlvl[:,int(padlt):-int(padrb),int(padlt):-int(padrb),:] # b,q,wh,ww,a -> b,H,W,a
        

        attn_mask = outputs_mask_curlvl.permute(0,3,1,2).repeat(1,numq,1,1) #b,H,W,a -> b,q,H,W
        
        local_attn_mask = torch.zeros((window_ids.shape[0],numq,window_ids.shape[2],window_ids.shape[3]),device=window_ids.device)#.to(window_ids)
        local_attn_mask = local_attn_mask.scatter_(dim=1,index=window_ids.long(),src=torch.ones_like(window_ids))
        local_attn_mask = (local_attn_mask[:,:,int(padlt):-int(padrb),int(padlt):-int(padrb)]==0).bool() # b,q,H,W
        local_attn_mask = local_attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) #bhead,q,HW


        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool() #小于0.5的是True，会被mask掉，认为小于0.5的是背景，大于0.5的是前景
        attn_mask_with_local = attn_mask*local_attn_mask
        attn_mask_with_local = attn_mask_with_local.detach()

        return attn_mask_with_local
        
        

    def forward_prediction_heads(self, output, mask_features, src, srcsize, level_index, fpn, nt_lang):
        '''full RSH: Include both main branch and no-target branch to simultaneously obtain the output mask and indicator.'''

        numq = output.shape[0]

        decoder_output = self.decoder_norm[level_index](output)
        decoder_output = decoder_output.transpose(0, 1) #100,bs,256 -> bs,100,256

        mask_embed = self.mask_embed[level_index](decoder_output) #bs,100,256 -> bs,100,256(maskdim)
        

        if level_index==0:
            padlt = 0
            padrb = 1

        else:
            padlt = 2**(level_index-1)
            padrb = 2**(level_index-1)

        '''main branch'''
        if fpn == False: # FPN additionally operates after the last layer
            mask_sw = None

            srcsize = srcsize[level_index]
            window_size = torch.div(srcsize[2], (torch.sqrt(torch.tensor(numq,device=output.device)).long()), rounding_mode='floor')+1
            
            src_sw = src[level_index].permute(1,2,0).reshape(srcsize[0],-1,srcsize[2],srcsize[3]) #sbc->bchw

            src_sw = F.pad(src_sw,(int(padlt),int(padrb),int(padlt),int(padrb)))

            _,_,paddedH,paddedW = src_sw.shape

            src_sw,_ = window_partition(src_sw.permute(0,2,3,1),window_size) #bchw->b,numq,wh,ww,c
            

            outputs_mask_curlvl = torch.einsum("bqac,bqhwc->bqhwa",mask_embed.unsqueeze(2),src_sw)
            #pooled_src = F.adaptive_avg_pool2d(src_sw.flatten(0,1).permute(0,3,1,2),(1,1)).squeeze(-1).squeeze(-1).reshape_as(mask_embed)
            #pooled_src = self.pooler_src_proj[level_index](pooled_src)
            outputs_mask_curlvl = window_reverse(outputs_mask_curlvl.flatten(0,1),window_size,paddedH,paddedW) # b,q,wh,ww,a -> b,H,W,a
            outputs_mask_curlvl = outputs_mask_curlvl[:,int(padlt):-int(padrb),int(padlt):-int(padrb),:] # b,q,wh,ww,a -> b,H,W,a
        else:
            window_size = (torch.div(mask_features.shape[2], (torch.sqrt(torch.tensor(numq,device=output.device)).long()), rounding_mode='floor')+1)

            mask_sw = F.pad(mask_features,(int(padlt),int(padrb),int(padlt),int(padrb)))

            _,_,paddedH,paddedW = mask_sw.shape
            mask_sw,_ = window_partition(mask_sw.permute(0,2,3,1),window_size)
            

            outputs_mask_curlvl = torch.einsum("bqac,bqhwc->bqhwa",mask_embed.unsqueeze(2),mask_sw)
            #pooled_src = F.adaptive_avg_pool2d(mask_sw.flatten(0,1).permute(0,3,1,2),(1,1)).squeeze(-1).squeeze(-1).reshape_as(mask_embed) #b,q,c

            #pooled_src = self.pooler_src_proj[level_index](pooled_src)

            outputs_mask_curlvl = window_reverse(outputs_mask_curlvl.flatten(0,1),window_size,paddedH,paddedW) # b,q,wh,ww,a -> b,H,W,a
            outputs_mask_curlvl = outputs_mask_curlvl[:,int(padlt):-int(padrb),int(padlt):-int(padrb),:] # b,q,wh,ww,a -> b,H,W,a

        '''no-target branch'''
        # nt_emb = self.nt_emb_proj[level_index](mask_embed)
        
        # nt_src_emb = torch.einsum('bqc,bqca -> bqa',nt_emb,pooled_src.unsqueeze(-1)).permute(0,2,1) #b,1,q

        # nt_lang_emb = self.nt_lang_emb_proj[level_index](nt_lang) # b,256,20->b,256,1
        
        # nt_lang_emb = torch.einsum('bca,bqc->baq',nt_lang_emb,nt_emb) #b,1,q

        # nt_label = torch.cat([nt_lang_emb,nt_src_emb],dim=-1)
        # nt_label = self.nt_embed[level_index](nt_label).squeeze(-2) #b,1,2


        outputs_mask_gathered = outputs_mask_curlvl.squeeze(-1)#torch.sum((outputs_mask_curlvl,gather),dim=1)
        
        return outputs_mask_gathered#, nt_label#, mask_sw
