import random
import tqdm
import numpy as np

import torch
import torch.optim as optim
from AutoRegressiveWrapper import AutoRegressiveWrapper

from long_short_transformer import LongShortTransformer
import Utils
import sys
import math
import os
from transformers import AutoTokenizer  # pip install transformers
import prep_text8

# ---------------developed by AM------------------------------------------ 
# Acknowledgement: some parts of code are adapted from Andrej Karpathy and Phil Wang
#-------------------------------------------------------------------------

# ------Architecture constants------------
DO_WORD_LEVEL_MODELING = False # set to false for character level, true for word
NUM_BATCHES = int(1e6)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 2
LEARNING_RATE = 3e-4
VALIDATE_EVERY  = 500
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LENGTH = 1024
RESUME_TRAINING = True # set to false to start training from beginning

EMBEDDING_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 8
WINDOW_SIZE = 128 # for short attention
SEGMENT_SIZE = 16 # for long attention
LONG_COMPRESSION_R = 4 # size for compressing a segment in long attention
USE_TOPK_CACHE = False  # to implement cache (using previous topkx3 attentive
                       # segments in expanded form)
#---------------------------
tokenizer_word = AutoTokenizer.from_pretrained("bert-base-cased",truncation=True, max_length=512) # for word level mmodeling
#tokenizer = AutoTokenizer.from_pretrained("bert-large-cased",truncation=True, max_length=1024)

#following commented functions are for character level modeling----------
def decode_token_char(token): # convert token to character
    return str(chr(max(32, token)))

def decode_tokens_char(tokens): # convert sequence of characters to tokens
    return ''.join(list(map(decode_token_char, tokens)))
#------------------------------------------------------------------------

def decode_tokens_word(tokens): # convert token to word - for word level modeling
    return tokenizer_word.decode(tokens)

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(mymodel):
    """
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in mymodel.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in mymodel.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9,0.95))
    return optimizer


def main():
    #prep_text8.prepare_text8() # prepare text8 data

    NUM_TOKENS = 256 # for character level modeling
    if DO_WORD_LEVEL_MODELING == True:
        NUM_TOKENS = 28996 # bert-base_cased for wikitext-103
    dim_head = int(EMBEDDING_SIZE/NUM_HEADS)
    longshort_model = LongShortTransformer(
        dim = EMBEDDING_SIZE, # embedding size - orig 512
        #num_unique_tokens = 28996, # for bert-base_cased for wikitext-103, 
        num_tokens = NUM_TOKENS,   
        # LongShortTransformer uses num_tokens instead of num_unique_tokens
        # it should be 256 for character level modeling
        depth = NUM_LAYERS, # LongShortTransformer calls num_layers as depth
        heads = NUM_HEADS, # orig 8
        dim_head = dim_head, # dimensionality of each head
        max_seq_len = SEQ_LENGTH,
        window_size = WINDOW_SIZE,  # for LSTransformer long attention
        segment_size = SEGMENT_SIZE,
        r=LONG_COMPRESSION_R, # compression size for each segment in the long attention case
        use_topk_cache = USE_TOPK_CACHE
    ).cuda()

    model = AutoRegressiveWrapper(longshort_model)
    model.cuda()
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")

    if DO_WORD_LEVEL_MODELING == True:
        train_loader, val_loader, test_loader, val_dataset = Utils.get_loaders_wikitext_103(tokenizer_word, SEQ_LENGTH, BATCH_SIZE)
    else: # char level modeling
        train_loader, val_loader, test_loader, val_dataset = Utils.get_loaders_enwiki8(SEQ_LENGTH, BATCH_SIZE)
        #train_loader, val_loader, test_loader, val_dataset = Utils.get_loaders_text8(SEQ_LENGTH, BATCH_SIZE)

    #optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)      # optimizer
    optim = configure_optimizers(model)

    # --------training---------
    if RESUME_TRAINING == False:
        start = 0
    else:
        checkpoint_data = torch.load('checkpoint/transAM_ENWK_Long_Short_178K_1.3.pt')
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        start = checkpoint_data['epoch']

    for i in tqdm.tqdm(range(start,NUM_BATCHES), mininterval = 10., desc = 'training'):
        model.train()
        total_loss = 0
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()
            print("loss computed..")
        if (i%100 == 0):
            print(f'training loss: {loss.item()} -- iteration = {i}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            total_len2 = 0
            total_loss2 = 0
            val_count = 1000  # number of validations to compute average BPC
            with torch.no_grad():
                for v in range(0,val_count):
                    loss = model(next(test_loader))
                    total_loss += loss.item()
                    loss_m = loss.mean()
                    total_loss2 += SEQ_LENGTH * loss_m.item() #loss.float().item() #seq_len
                    total_len2 += SEQ_LENGTH
                print(f'----------validation loss: {total_loss/val_count}')
                print(f'Perplexity : {math.exp(total_loss/val_count)}, BPC: {total_loss/val_count*np.log2(2.7173)}')
                bpc2 = (total_loss2/total_len2)/math.log(2)
                print("BPC 2 = ", bpc2)
                total_loss = 0
        if i % GENERATE_EVERY == 0:  
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            if DO_WORD_LEVEL_MODELING == True:
                input_start_sequence = decode_tokens_word(inp)
            else:
                input_start_sequence = decode_tokens_char(inp)
            print("----------start input------------------")
            print(f'%s \n\n', (input_start_sequence))
            print("----------end of start input-----------")
            sample = model.generate(inp, GENERATE_LENGTH)
            if DO_WORD_LEVEL_MODELING == True:
                output_str = decode_tokens_word(sample)
            else:
                output_str = decode_tokens_char(sample)
            print("----------generated output-------------")
            print(output_str)
            print("----------end generated output---------")
            
            # ---------save the latest model---------
            print("----------saving model-----------------")
            checkpoint_data = {
            'epoch': i,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
            }
            ckpt_path = os.path.join("checkpoint/transAM_WK_model.pt")
            torch.save(checkpoint_data, ckpt_path)
            # revert model to training mode
            model.train()

if __name__ == "__main__":
    sys.exit(int(main() or 0))
