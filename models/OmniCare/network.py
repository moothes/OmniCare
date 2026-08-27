import torch
import numpy as np

import torch.nn as nn
import random
from torch.nn import functional as F
import math
import warnings
from .vit import Transformer

from conch.open_clip_custom import create_model_from_pretrained, create_model_from_pretrained, tokenize, get_tokenizer
from transformers import AutoModel

warnings.filterwarnings("ignore")



class Attention(nn.Module):
    def __init__(self, chan_nheads, resolution, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dim = dim
        self.resolution = resolution
        pixel_no = int(resolution[0] * resolution[1])
        self.pixel_no = pixel_no

        self.chan_nheads = chan_nheads
        chan_head_dim = self.pixel_no // self.chan_nheads
        self.chan_scale = chan_head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        raw_spa_attn = (q @ k.transpose(-2, -1))
        attn = raw_spa_attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        raw_spa_attn = raw_spa_attn, attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, task_no+1+HxW, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        raw_attn = [raw_spa_attn]

        return x, raw_attn

class LoraBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, rank=6):
        super().__init__()
        self.W = nn.Linear(in_channels, rank)
        self.M = nn.Linear(rank, out_channels)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)
    
    def forward(self, x):
        x = self.W(x)
        x = self.M(x)
        return x

class LoraBlock_trans(nn.Module):
    def __init__(self, dims, kernel_size=1, rank=6):
        super().__init__()
        #self.W = nn.Linear(in_channels, rank)
        #self.M = nn.Linear(rank, out_channels)
        self.cls = nn.Parameter(torch.randn(1, 1, dims), requires_grad=True)
        self.trans = Transformer(dim=dims, depth=2, heads=4, dim_head=rank, mlp_dim=rank)

    def init_weights(self):
        #nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        #nn.init.zeros_(self.W.bias)
        #nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        #nn.init.zeros_(self.M.bias)
        pass
    
    def forward(self, x):
        xin = torch.concat([self.cls, x], dim=1)#.unsqueeze(1)
        xout = self.trans(xin)
        return xout


class Block(nn.Module):

    def __init__(self, chan_nheads, resolution, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(chan_nheads, resolution, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x_attn, attn_weight = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weight

class SpatialAtt(nn.Module):
    def __init__(self, dim, dim_out, im_size, with_feat):
        super().__init__()
        self.conv1 = nn.Linear(dim, dim_out)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Linear(im_size, dim)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Linear(dim, dim_out)
        self.conv3 = nn.Linear(dim_out, dim_out)
        self.with_feat = with_feat
        if with_feat:
            self.feat_linear = nn.Linear(dim_out *2 , dim_out *2)
    
    def forward(self, x, route_feat=None):
        #n, c = x.shape
        feat = self.conv1(x)
        feat = self.ln(feat)
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = self.convsp(x)
        feat_sp = self.ln_sp(feat_sp)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)
        
        #n, c = feat.shape
        #feat = torch.mean(feat.reshape(n, c), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)

        return feat

class MOEBlock(nn.Module):
    def __init__(self, final_embed_dim, num_task, kernel_size=1, with_feat=False, rank_list=[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]):
        super().__init__()
        self.desert_k = 6
        self.num_lora = len(rank_list)
        self.lora_list_1 = nn.ModuleList()
        for i in range(self.num_lora):
            self.lora_list_1.append(LoraBlock_trans(final_embed_dim, kernel_size=kernel_size, rank=rank_list[i]))
            self.lora_list_1[-1].init_weights()
        self.conv1_cls = nn.Parameter(torch.randn(1, final_embed_dim), requires_grad=True)
        self.conv1 = Transformer(dim=final_embed_dim, depth=2, heads=4, dim_head=128, mlp_dim=128)
        self.conv2 = nn.ModuleList()
        self.share_conv = LoraBlock_trans(final_embed_dim, kernel_size=kernel_size, rank=64)
        self.bn_all = nn.ModuleList()
        for idx in range(num_task):
            self.conv2.append(LoraBlock_trans(final_embed_dim, kernel_size=kernel_size, rank=64))
            self.bn_all.append(nn.LayerNorm(final_embed_dim))
        self.router_1 = nn.ModuleList() 
        self.pre_softmax = False
        for idx in range(num_task):
            self.router_1.append(nn.ModuleList([SpatialAtt(final_embed_dim, final_embed_dim // 4, im_size=final_embed_dim, with_feat=with_feat), nn.Linear(final_embed_dim // 2, self.num_lora * 2 + 1)]))
        
    def forward(self, x, task, route_feat_in=None, phase='train'):
        xin = torch.concat([self.conv1_cls, *x], dim=0).unsqueeze(0) # (n, 5, d)
        xout = self.conv1(xin) # (n, 5, d)
        out, xfeat = xout[:, :1], xout[:, 1:] # (n, 1, d) (n, 4, d)
        n = out.shape[0]
        route_feat = self.router_1[task][0](out.squeeze(1), route_feat_in)
        prob_all = self.router_1[task][1](route_feat).unsqueeze(2)
        prob_lora, prob_mix = prob_all[:, :self.num_lora * 2], prob_all[:, self.num_lora * 2:] # (n, 30, 1), (n, 1, 1)
        route_1_raw, stdev_1 = prob_lora.chunk(2, dim=1)  # (n, 15, 1)*2
        if phase == 'train':
            noise = torch.randn_like(route_1_raw) * stdev_1
        else:
            noise = 0
        if self.pre_softmax:
            route_1_raw = route_1_raw + noise
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            for j in range(n):
                for i in range(self.desert_k):
                    route_1_raw[j, route_1_indice[j, i].reshape(-1)] = -1e10
            route_1 = torch.softmax(route_1_raw, dim=1)
        else:
            route_1_raw = torch.softmax(route_1_raw + noise, dim=1)
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            route_1 = route_1_raw.clone()
            for j in range(n):
                route_1[j, route_1_indice[j]] = 0
        lora_out_1 = []
        for i in range(self.num_lora):
            lora_out_1.append(self.lora_list_1[i](xfeat)[:, :1]) # n, 1, c
        lora_out_1 = torch.cat(lora_out_1, dim=1)
        lora_out_1 = torch.sum(lora_out_1 * route_1, dim=1)
        moe_out = self.bn_all[task](lora_out_1)
        spec_out = self.conv2[task](xfeat) * prob_mix[:, 0]
        share_out = self.share_conv(xfeat)
        return [moe_out, spec_out[:, 0], share_out[:, 0]], route_feat, route_1

def SNN_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.SELU(), nn.AlphaDropout(p=dropout, inplace=False))
    
def MLP_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.Dropout(dropout), nn.ReLU())
    #return nn.Sequential(nn.Linear(dim1, dim2), nn.LayerNorm(dim2), nn.ReLU())
    #return nn.Sequential(nn.Linear(dim1, dim2), nn.Dropout(dropout), nn.ELU())

def conv1d_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Conv1d(dim1, dim2, 1), nn.InstanceNorm1d(dim2), nn.ReLU())
    #return nn.Sequential(nn.Conv1d(dim1, dim2, 1), nn.Dropout(dropout), nn.ELU())
    
def conv2d_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Conv2d(dim1, dim2, 1), nn.InstanceNorm2d(dim2), nn.ReLU())
    #return nn.Sequential(nn.Conv2d(dim1, dim2, 1), nn.Dropout(dropout), nn.ELU())

class ABMIL(nn.Module):
    def __init__(self, path_dim=1024, feat_dim=64):
        super().__init__()
        self.feature = MLP_Block(path_dim, feat_dim)
        self.attention = nn.Sequential(nn.Linear(feat_dim, feat_dim//4), nn.Dropout(0.2), nn.ReLU(), nn.Linear(feat_dim//4, 1))
        
    def forward(self, x_path):
        feature = self.feature(x_path)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        return M, A#.unsqueeze(0)

class Transformer_block(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim), requires_grad=True)
        self.trans = Transformer(dim=feat_dim, depth=2, heads=4, dim_head=64, mlp_dim=64)
        
    def forward(self, x):
        feat = torch.concat([self.cls_token, x], dim=1)
        feat = self.trans(feat)
        return feat[:, 0], feat[:, 1:]

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.stage = args.stage

        self.feat_dim = 256
        path_dim = 2560
        
        self.modal_ratio = 0.1
        self.mask_ratio = 0.05
        
        ### Modality-specific embeddings
        
        # Cancer Embedding
        self.cancer_token = nn.Parameter(torch.randn(args.num_subsets, 1, self.feat_dim), requires_grad=True)
        
        # Pathology representation using ABMIL
        self.abmil_path = ABMIL(path_dim, self.feat_dim)
        self.path_holder = MLP_Block(self.feat_dim, self.feat_dim)

        # Pathology representation using ABMIL
        self.abmil_ihc = ABMIL(path_dim, self.feat_dim)
        self.ihc_holder = MLP_Block(self.feat_dim, self.feat_dim)

        # Genomic representation using MLP # 20245 for tcga
        self.omic_snn = nn.Sequential(MLP_Block(args.gene_length, self.feat_dim*2), MLP_Block(self.feat_dim*2, self.feat_dim))
        self.omic_holder = MLP_Block(self.feat_dim, self.feat_dim)

        # Report representation using MLP
        self.text_encoder, self.preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="Your token")
        self.text_snn = nn.Sequential(MLP_Block(512, self.feat_dim*2), MLP_Block(self.feat_dim*2, self.feat_dim))
        self.text_holder = MLP_Block(self.feat_dim, self.feat_dim)
        
        # Clinical representation using MLP
        self.clin_encoder = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
        self.clin_snn = nn.Sequential(MLP_Block(1024, self.feat_dim*2), MLP_Block(self.feat_dim*2, self.feat_dim))
        self.clin_holder = MLP_Block(self.feat_dim, self.feat_dim)

        ### Multimodal fusion
        self.moe_cls_token = MLP_Block(self.feat_dim, self.feat_dim)
        self.moe_block = MOEBlock(self.feat_dim, args.num_subsets, kernel_size=3, with_feat=False)
        self.transformer = Transformer_block(self.feat_dim)
        
        ### Final prediction
        #self.classifier = nn.Sequential(nn.Linear(self.feat_dim, args.n_classes))
        
        ### Prediction head for pretraining
        if self.stage == 'pretrain':
            self.pretrain_head = nn.ModuleDict()
            #head_list = ['os', 'diag', 'text', 'clin', 'ihc', 'he', 'gene']
            self.pretrain_head['os'] = nn.Linear(self.feat_dim, 5)
            self.pretrain_head['diag'] = nn.Linear(self.feat_dim, 32)
            # Reconstruction
            self.pretrain_head['text'] = nn.Linear(self.feat_dim, 512)
            self.pretrain_head['clin'] = nn.Linear(self.feat_dim, 1024)
            self.pretrain_head['ihc'] = nn.Linear(self.feat_dim, self.feat_dim)
            self.pretrain_head['he'] = nn.Linear(self.feat_dim, self.feat_dim)
            self.pretrain_head['gene'] = nn.Linear(self.feat_dim, 768)
        else:            
            self.classifier = nn.Sequential(nn.Linear(self.feat_dim, args.n_classes))

          
    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
  
    def forward(self, data_dict, phase='test'):
        all_out = []
        for kwargs in data_dict:
            out_dict = {}
            cid = kwargs['cid']
            
            rec_he_feat = None
            rec_ihc_feat = None
            
            feats = []
            #cid = cancer_id[batch_idx]
            
            # cancer-specific embedding
            cpt = self.cancer_token[cid]

            available_modal = list(kwargs.keys())
            if 'cid' in available_modal:
                available_modal.remove('cid')
            if 'clin_attention_mask' in available_modal:
                available_modal.remove('clin_attention_mask')
                
            if phase == 'pretrain' and len(available_modal) > 2 and torch.rand(1).item() < self.modal_ratio:
                modal = random.choice(available_modal)
                available_modal.remove(modal)
                #print(kwargs.keys(), available_modal)
            
            if 'path' in available_modal:
                x_path = kwargs['path']
                if phase == 'pretrain' and torch.rand(1).item() < self.mask_ratio:
                    #print(x_path.shape)
                    indices = torch.randperm(len(x_path))[:2000]
                    x_path_new = x_path[indices]
                else:
                    x_path_new = x_path
                #print(x_path.shape, x_path_new.shape)
                path_feat, path_att = self.abmil_path(x_path_new)
                feats.append(path_feat)
                out_dict['path_att'] = path_att
            else:
                path_feat = self.path_holder(cpt)
            
            if phase == 'pretrain' and 'path' in kwargs.keys():
                x_path = kwargs['path']
                with torch.no_grad():
                    rec_he_feat, _ = self.abmil_path(x_path)
                
            if 'ihc' in available_modal:
                x_path = kwargs['ihc']
                if phase == 'pretrain' and torch.rand(1).item() < self.mask_ratio:
                    indices = torch.randperm(len(x_path))[:2000]
                    x_path_new = x_path[indices]
                else:
                    x_path_new = x_path
                #print(x_path.shape, x_path_new.shape)
                ihc_feat, _ = self.abmil_ihc(x_path_new)
                feats.append(ihc_feat)
            else:
                ihc_feat = self.ihc_holder(cpt)
                
            if 'ihc' in kwargs.keys():
                x_path = kwargs['ihc']
                with torch.no_grad():
                    rec_ihc_feat = self.abmil_ihc(x_path)
                
            if 'gene' in available_modal:
                x_omic = kwargs['gene']
                #print('Gene shape: ', x_omic.shape)
                omic_feat = self.omic_snn(x_omic)#.unsqueeze(0)
                feats.append(omic_feat)
            else:
                omic_feat = self.omic_holder(cpt)
                
            if 'text' in kwargs.keys():
                x_text = kwargs['text']
                #print(x_text.shape)
                if phase == 'pretrain' and torch.rand(1).item() < self.mask_ratio:
                    mask = (torch.rand([1, 128]) < 0.2).float().cuda()
                    x_text = (x_text * mask).long()
                #print('Text shape: ', x_text.shape)
                with torch.no_grad():
                    text_tokens = self.text_encoder.encode_text(x_text)
                text_feat = self.text_snn(text_tokens)
                feats.append(text_feat)
            else:
                text_feat = self.text_holder(cpt)
        
            if 'clin' in kwargs.keys():
                #x_clin = kwargs['clin']
                clin_input_ids = kwargs['input_ids']
                if phase == 'pretrain' and torch.rand(1).item() < self.mask_ratio:
                    mask = (torch.rand([1, 1024]) < 0.2).float().cuda()
                    clin_input_ids = clin_input_ids * mask
                #print('Clin shape: ', clin_input_ids.shape)
                clin_attention_mask = kwargs['attention_mask']
                clin_dict = {'clin_input_ids': clin_input_ids, 'clin_attention_mask': clin_attention_mask}
                with torch.no_grad():
                    outputs = self.clin_encoder(**clin_dict)
                    embeddings = self.last_token_pool(outputs.last_hidden_state, clin_attention_mask)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                clin_feat = self.clin_snn(embeddings)
                feats.append(clin_feat)
            else:
                clin_feat = self.clin_holder(cpt)
        
            #all_feats.append(feats)
            feat = torch.stack([self.moe_cls_token(cpt), path_feat, ihc_feat, omic_feat, text_feat, clin_feat], dim=1)
            knows, route_feat, route_1 = self.moe_block(feat, cid, phase=phase)
            
            feat = torch.stack(knows, dim=1)
            cls_out, feat = self.transformer(feat)
            #feat = self.classifier(cls_out)
            #logits.append(cls_out)
            
            out_dict['router'] = route_1
            #print(route_1.shape)
            if self.stage == 'pretrain':
                for key, head in self.pretrain_head.items():
                    out_dict[key] = head(cls_out)
            
                out_dict['pred'] = cls_out
                out_dict['modality_feats'] = feats
                out_dict['mmfeat'] = cls_out
                out_dict['reco_he'] = rec_he_feat
                out_dict['reco_ihc'] = rec_ihc_feat
            else:
                pred = self.classifier(cls_out)
                out_dict['pred'] = pred
                
            
            all_out.append(out_dict)
        #logits = torch.concat(logits, dim=0)
        
        #out_dict['pred'] = logits
        #out_dict['modality_feats'] = all_feats
        #out_dict['mmfeat'] = cls_out
        
        return all_out
