import torch
from torch import nn
import torch.nn.functional as F

# ---------top k from logits, logits = layer before softmax
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1]) # top 25
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf')) # fill probs=[1,256] with -inf
    probs.scatter_(1, ind, val) # 1=dim, it will fill probs with val at ind location
    return probs                # i.e., top 25 locations will have values, rest -inf

class AutoRegressiveWrapper(nn.Module):
    def __init__(self, net, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.model = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_thres = 0.9):
        self.model.eval()
        device = start_tokens.device  # start tokens is the seed set of characters
        num_dims = len(start_tokens.shape) # 1, e.g., start_tokens = 1024

        if num_dims == 1:
            start_tokens = start_tokens[None, :]    # 1024 --> (1, 1024); [None, :] is an alias for np.newaxis, it will add one more dimension

        b, t = start_tokens.shape   # b=1, e.g., t=1024. In generation, batch=1
        prev_out = start_tokens  # e.g., [1x1024]
        for _ in range(seq_len):    # seq_len = e.g., 1024
            x = prev_out[:, -self.max_seq_len:]  # x=(1, 1024); max_seq_len = 1024
            #self.model.set_causal(False) - causes generated output to be not proper
            logits = self.model(x)[:, -1, :] # x = [1x1024], logits = [1x256]

            filtered_logits = top_k(logits, thres = filter_thres)  # filtered_logits = (1x256)
            # top 10% logits will be kept, others changed to -inf.
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            predicted_char_token = torch.multinomial(probs, 1) # (1x1)
            out = torch.cat((prev_out, predicted_char_token), dim=-1)  # (1 x 1025)
            prev_out = out
            if eos_token is not None and (predicted_char_token == eos_token).all():
                break

        out = out[:, t:]    # generated output sequence after the start sequence, e.g., (1x512)
        if num_dims == 1:
            out = out.squeeze(0)
        #self.model.set_causal(True) - causes generated output to be not proper
        return out

 
    def forward(self, x):
        xi = x[:, :-1] # x is input of size seq_len+1, :-1 will make it seq_len 
        xo = x[:, 1:]  # expected output in training is shifted one to the right
        out = self.model(xi)
        logits_reorg = out.view(-1, out.size(-1))
        targets_reorg = xo.reshape(-1)
        loss = F.cross_entropy(logits_reorg, targets_reorg, ignore_index=-1)
        return loss
