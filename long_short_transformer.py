from math import gcd, ceil
import functools

import torch
from torch import nn, einsum
import torch.nn.functional as F

from Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb

from einops import rearrange, repeat
import numpy as np
import matplotlib.pyplot as plt


# -----------helper functions---------------
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def lcm(*numbers):  # least common multiple, e.g., 12, 15, 75 => 300
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def pad_to_multiple(tensor, multiple, dim = -1, value = 0): 
    seqlen = tensor.shape[dim]  
    m = seqlen / multiple 
    if m.is_integer():
        return tensor
    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
   
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

   
def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]  
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

#----------------------end helper functions----------------


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class LongShortAttention(nn.Module):
    def __init__(
        self,
        *,
        dim, # embedding size
        heads = 8,
        dim_head = 64,
        causal = True,
        sequence_length = 1024,
        window_size = 128,
        pos_emb = None,
        segment_size = 16,
        r = 1, # projection of each segment
        use_topk_cache=False,
        topk=5,
        dropout = 0.
    ):
        super().__init__()
        assert not (causal and r >= segment_size), 'r should be less than segment size, if autoregressive'

        inner_dim = heads * dim_head  
        self.scale = dim_head ** -0.5  

        self.heads = heads
        self.causal = causal

        self.window_size = window_size  
        self.segment_size = segment_size  
        self.pad_to_multiple = window_size if not causal else lcm(window_size, segment_size)  
        self.to_dynamic_proj = nn.Linear(dim_head, r, bias = False)  
        self.local_norm = nn.LayerNorm(dim_head)  
        self.global_norm = nn.LayerNorm(dim_head)  

        self.pos_emb = default(pos_emb, RotaryEmbedding(dim_head))

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False) 
        self.to_kv = nn.Linear(dim, inner_dim, bias = False)    
        self.to_out = nn.Linear(inner_dim, dim) 
        self.r = r 
        self.use_use_topk_cache = use_topk_cache
        self.topk = topk
        self.sequence_length = sequence_length

    # ---------for implementing topk cache
    def fill_unique_values(self, input_tensor, k, max_segments): 
        # max_segments = 63 for 1024 seq len, seg size of 16
        # k = topk, e.g. 5, 
        # we obtain k*3 segments , before,current,next, for k=5, there will be 15 segments   
        max_seg_plus_one = max_segments + 1
        results = []
        kn = k * 3
        tensor_i = torch.arange(0, kn).cuda() 
        for i in range(input_tensor.shape[-2]): 
            unique_tensor = torch.unique_consecutive(input_tensor[i],dim=-1) 
            # may become less size as we remove duplicates, will need to fill
            print(unique_tensor)
            for k in range(0,len(unique_tensor)): 
                if (i == 21):
                    aa = 5
                if unique_tensor[k] >= (i%max_seg_plus_one):  # autoregressive, cannot look into future segment
                    unique_tensor[k] = (i%max_seg_plus_one)-1
                if unique_tensor[k] == -1: # prev of segment 0 may be received as -1
                    unique_tensor[k] = 0
                if unique_tensor[k] >= max_segments: # next of max_segment 
                    unique_tensor[k] = max_segments-1
            min_value = unique_tensor.min()
            if i < kn:  
                min_value = 0
            max_value = unique_tensor.max()
            if min_value > max_value:
                min_value = 0
        
            range_values = torch.arange(min_value, max_value + 1).cuda()
            present_mask = torch.isin(range_values, unique_tensor)
            filled_tensor = range_values[present_mask]
            if (i%max_seg_plus_one) < kn+1:
                res = tensor_i
            else:
                if len(filled_tensor) > kn:
                    filled_tensor = filled_tensor[:kn]
                # If filled tensor has less values than kn, fill it
                elif len(filled_tensor) < kn:
                    padding_needed = kn - len(filled_tensor)
                    tensor_ii = torch.arange(0,(i%max_seg_plus_one)).cuda()
                    padding_values=tensor_ii[(tensor_ii != filled_tensor.view(-1, 1)).all(dim=0)]
                    filled_tensor = torch.cat([filled_tensor,padding_values[len(padding_values)-padding_needed:len(padding_values)]]) # Only last padding_needed
                    res, _ = torch.sort(filled_tensor)
                    if len(res) > kn:
                        res = res[len(res)-kn:len(res)]
            results.append(res)
            res_total = torch.stack(results)
        return res_total

    def forward(self, x, mask = None): 
        #if mask is not None: # mask is None only when generate is triggered
 
        b, n, *_, h, device, causal, w, s = *x.shape, self.heads, x.device, self.causal, self.window_size, self.segment_size
        
        # pad input sequence to multiples of window size (or window size and segment length if causal)
        # ---padding by zeros, if the sequence length is not a multiple of
        # window size, e.g., if window size =120, then 120x9 = 1080, which means
        # we need padding of 56 to if the seqlength was 1024  - by AM---

        x = pad_to_multiple(x, self.pad_to_multiple, dim = -2, value = 0.)  # 1024, 128

        # derive from variables
        padded_len = x.shape[-2]    
        windows = padded_len // w   
        is_padded = padded_len != n

        mask_value = -torch.finfo(x.dtype).max

        # handle mask if padding was needed and mask was not given

        if is_padded:
            mask = default(mask, torch.ones((b, n), device = device).bool()) 
            mask = pad_to_multiple(mask, w, dim = -1, value = False)
            

        # get queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))  

        # get sequence range, for calculating mask
        seq_range = torch.arange(padded_len, device = device)

        # split heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv) 

        # rotary embedding
        if exists(self.pos_emb):
            rotary_emb = self.pos_emb(seq_range, cache_key = padded_len) 
            rotary_emb = rearrange(rotary_emb, 'n d -> () n d') 
            q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        # scale queries

        q = q * self.scale

        # get local queries and keys similarity scores
        window_fn = lambda t: rearrange(t, 'b (w n) d -> b w n d', n = w)  
        lq, lkv = map(window_fn, (q, kv))  

        lookaround_kwargs = {'backward': 1, 'forward': (0 if causal else 1)}
        lkv = look_around(lkv, **lookaround_kwargs)   
        # [-1,0], [0,1], [1,2], ..... [6,7],  each pair [128,2] i.e., 256
        # sliding window is implemented here differently from paper
        # but it also makes sense. Overlap is w only on left side

        lkv = self.local_norm(lkv)
        lsim = einsum('b w i d, b w j d -> b w i j', lq, lkv) 
        # prepare global key / values
        if self.causal:
            # autoregressive global attention is handled in segments
            # later on, these segments are carefully masked to prevent leakage

            gkv = rearrange(kv, 'b (n s) d -> b n s d', s = s)  
            pkv = self.to_dynamic_proj(gkv)   
            # project embedding to r - concept borrowed from performer, 
            # then it will be used to further reduce the seq length

            if exists(mask): # it is triggered when generate triggers
                pmask = repeat(mask, 'b (n s) -> (b h) n s', s = s, h = h) 
                pkv.masked_fill_(~pmask[..., None], mask_value) 

            pkv = pkv.softmax(dim = -2)  

            gkv = einsum('b n s d, b n s r -> b n r d', gkv, pkv)     
            gkv = rearrange(gkv, 'b n r d -> b (n r) d')  
            aa =5
        else:
            # equation (3) in the paper
            pkv = self.to_dynamic_proj(kv)
            if exists(mask):
                pkv.masked_fill_(~mask[..., None], mask_value)

            pkv = pkv.softmax(dim = -2)
            gkv = einsum('b n d, b n r -> b r d', kv, pkv)

        # calculate global queries and keys similarity scores
        gkv = self.global_norm(gkv) 
        gsim = einsum('b n d, b r d -> b n r', q, gkv)  

        #-----by AM - section for segment based cache implementation----
        if self.use_use_topk_cache == True:
            gsim2 = rearrange(gsim, 'b l (c r) -> b l c r', r=self.r)    
            gsim2n = torch.norm(gsim2, dim=-1) 
            gsim2n16 = rearrange(gsim2n, 'b (m n) c -> b m n c', n=16)  
            gg = torch.mean(gsim2n16, dim=2)    
            values, indexes = torch.topk(gg, 5, dim=-1) 
        
            next_indexes = indexes+1    
            prev_indexes = indexes-1    
            filled_indexes = torch.stack([prev_indexes, indexes,next_indexes], dim=-1)  
            bat = filled_indexes.shape[0] # batch size = 32
            full_indexes = rearrange(filled_indexes, 'b m n c -> (b m) (n c)') 
            max_segments = int(self.sequence_length/self.segment_size) - 1 
            unique_topk_segments3 = self.fill_unique_values(full_indexes,self.topk, max_segments) #topk=5, max_segments=63, future segments are set to max

            kv_seg = rearrange(kv,'b (i j) k->b i j k', j=16) 
            
            kv_seg2 = rearrange(kv_seg,'b i j k->(b i) j k') 
            q2 = rearrange(q,'b (n s) e->b n s e', s=16)
            q3 = rearrange(q2,'b n s e-> (b n) s e') 
         
            res = torch.zeros(q3.size(0),16,240).cuda()

            
            vcache = torch.zeros(256,240,64).cuda() # to improve later
            ii = 0 
            for i in range(0,kv_seg2.size(0)):  # 2048 times
                ks = kv_seg2[unique_topk_segments3[i]]
                
                ks2 = rearrange(ks,'i j k->(i j) k') 
                
                vcache[ii] = ks2
                if (i+1) % 8 == 0: 
                    ii = ii + 1
                gc = torch.einsum('n e,m e->n m',[q3[i],ks2]) 
                res[i] = gc
    
            gcache = rearrange(res,'(b j) k o->b (j k) o', b=bat)
           
        #----------------end cache handling--------------------------------
 
        # concat values together (same as keys)

        gkv = repeat(gkv, 'b r d -> b w r d', w = windows)  
        
        if self.use_use_topk_cache == True:
            vcache = rearrange(vcache,'(i h) j k->i h j k',h=self.heads)
            v = torch.cat((gkv, lkv,vcache), dim = -2)  
        else:
            v = torch.cat((gkv, lkv), dim = -2)  

        # masking
        buckets, i, j = lsim.shape[-3:]  

        if exists(mask): 
            mask = repeat(mask, 'b (w n) -> (b h) w n', n = w, h = h) 
            mask = look_around(mask, pad_value = False, **lookaround_kwargs)
            mask = rearrange(mask, 'b w n -> b w () n') 
            lsim.masked_fill_(~mask, mask_value)

        # mask out padding
        seq_range_windowed = rearrange(seq_range, '(w n) -> () w n', w = windows)   
        pad_mask = look_around(seq_range_windowed, pad_value = -1, **lookaround_kwargs) == -1  
        lsim.masked_fill_(pad_mask[:, :, None], mask_value) 

        # calculate causal masking for both global and local

        if self.causal:
            g_range = rearrange(seq_range, '(n s) -> n s', s = s)   
            g_range_max = g_range.amax(dim = -1)    

            
            g_mask = seq_range[:, None] > g_range_max[None, :] 
            g_mask = repeat(g_mask, 'm n -> m (n repeat)', repeat=self.r) 

            g_mask = rearrange(g_mask, 'i j -> () i j')
            gsim.masked_fill_(~g_mask, mask_value)

            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = repeat(causal_mask, 'i j -> () u i j', u = buckets)
            lsim.masked_fill_(causal_mask, mask_value)

        # concat local and global similarities together to ready for attention
        
        gsim = rearrange(gsim, 'b (w n) r -> b w n r', w = windows) 
        if self.use_use_topk_cache == True:
            gcache = rearrange(gcache,'b (w n) r-> b w n r', w=windows) # AM
            sim = torch.cat((gsim, lsim, gcache), dim = -1)
        else:
            sim = torch.cat((gsim, lsim), dim = -1) # original

        # final attention
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values (same as keys, since tied) and project out
        out = einsum('b w i j, b w j d -> b w i d', attn, v)
        out = rearrange(out, '(b h) w n d -> b (w n) (h d)', h = h)
        out = out[:, :n]
        return self.to_out(out)


class LongShortTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # for output size determination
        dim,   # embedding
        depth, # number of layers
        heads = 8,
        dim_head = 64, # dimensionality of each head
        max_seq_len,
        window_size = 128,
        segment_size = 16,
        r = None,
        use_topk_cache = False,
        causal = True,
        ff_mult = 4,  # expand feedforward by 4 times then reduce back to embedding size
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        pos_emb = RotaryEmbedding(dim_head)

        # handle autoregressive default variables differently
        # specifically, segments are only used for autoregressive case
        # r is the projected r << n in the non-autoregressive case, and the projected r per segment for the autoregressive case

        segment_size = default(segment_size, 16 if causal else None)
        r = default(r, 1 if causal else 128)  # orig


        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LongShortAttention(dim = dim, heads = heads, dim_head = dim_head, sequence_length = max_seq_len, window_size = window_size, causal = causal, pos_emb = pos_emb, segment_size = segment_size, r = r, use_topk_cache=use_topk_cache, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        x = self.token_emb(x)
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x   
        return self.to_logits(x)
