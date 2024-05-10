# -*- coding: utf-8 -*-
# Time    : 2024/5/10
# By      : Yang

import numpy as np
import torch
import torch.nn as nn


def positional_encode(max_seq_len, d_model):
    """ positional encode
    Args:
        max_seq_len: the max length of the all source/target seqeunce.
        d_model: the feature dimension length of seqeunce.
    """
    pos_enc = torch.zeros((max_seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        pos_enc[:, i] = f(torch.arange(0, max_seq_len) / np.power(1e4, 2 * i / d_model))
    
    # pso_enc: [seq_len, d_model]
    return pos_enc.float()


def get_enc_attn_mask(batch_size, max_src_len, eas_feat_len, device):
    """ encoder mask
    Args:
        batch_size: batch-size.
        max_src_len: the max length of the all source seqeunce.
        eas_feat_len: the feature dimension length of each source seqeunce.
        device: cuda or cpu.
    """
    enc_mask = torch.zeros((batch_size, max_src_len, max_src_len), device=device)
    for i in range(batch_size):
        enc_mask[1, :, :eas_feat_len[i]] = 0
    
    return enc_mask.to(torch.bool)


def get_subsequent_mask(batch_size, max_tgt_len, device):
    """ decoder mask
    Args:
        batch_size: batch-size.
        max_tgt_len: the max length of the all target seqeunce.
        device: cuda or cpu.
    """
    #  Generate the upper triangular matrix
    dec_sub_mask = torch.triu(torch.ones((batch_size, max_tgt_len, max_tgt_len), device=device), diagonal=1)
    
    # [batch_size, max_tgt_len, max_tgt_len]
    return dec_sub_mask.to(torch.bool)


def get_enc_dec_mask(batch_size, max_src_len, eas_feat_len, max_tgt_len, device):
    ''' encoder decoder mask
    Args:
        batch_size: batch-size.
        max_src_len: the max length of the all source seqeunce.
        eas_feat_len: the feature dimension length of each source seqeunce.
        max_tgt_len: the max length of the all target seqeunce.
        device: cuda or cpu.
    '''
    ed_attn_mask = torch.zeros((batch_size, max_tgt_len, max_src_len), device=device)  # (b, seq_q, seq_k)
    for i in range(batch_size):
        ed_attn_mask[i, :, eas_feat_len[i]:] = 1
    
    # [batch_size, max_tgt_len, max_src_len]
    return ed_attn_mask.to(torch.bool)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, p=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p)
        
        self.norm = nn.LayerNorm(d_model)
        
        # linear projections [d_model,d_model]
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
        # Normalization
        # References: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.fc.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
    
    def forward(self, input_Q, input_K, input_V, attn_mask):
        redidual = input_Q
        
        batch_size = input_Q.size(0)
        
        len_q, len_k = input_Q.size(1), input_K.size(1)
        
        # multi head split, d_k/d_v = d_model // n_heads
        # (b, seq_len, d_model) > (b, len_qkv, n_heads, d_kv) > (b, n_heads, len_qkv, d_kv)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_Q(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_Q(input_V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # pre-process mask, attn_mask : [batch_size, n_heads, seq_len, seq_len]
        if attn_mask is not None:
            assert attn_mask.size() == (batch_size, len_q, len_k)
            # broadcast
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            attn_mask = attn_mask.bool()
        
        # Scaled Dot Product Attention
        # calculate attention weight scores:  scores =  softmax(Q*K^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        # add attn_mask for attention weight score
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        
        # use softmax
        attn_weight = torch.softmax(scores, dim=-1)
        attn_weight = self.dropout(attn_weight)
        
        # calculate output: res = softmax(Q*K^T / sqrt(d_k)) * V
        # output.shape = (batch_size, n_heads, seq_len, d_v)
        output = torch.matmul(attn_weight, V)
        
        # multi_head attention merge
        # (batch_size, n_heads, seq_len, d_v) - (batch_size, seq_len, n_heads, d_v) - (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.d_v * self.n_heads)
        output = self.fc(output)
        
        # residual output
        return self.norm(redidual + output), attn_weight


class PosiWiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, p=0.):
        super(PosiWiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        residual = inputs
        
        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        # residual output
        return self.norm(out + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, pwffn_p, attn_p):
        """
        Args:
            d_model: input dimension
            n_heads: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            pwffn_p: dropout ratio of Position Wise Feed Forward Net
            attn_p: dropout ratio of attention module
        """
        assert d_model % n_heads == 0
        # dimension of each attention head
        d_qkv = d_model // n_heads
        super(EncoderLayer, self).__init__()
        
        # MultiHeadAttention, self-attention
        self.enc_self_attn = MultiHeadAttention(d_qkv, d_qkv, d_model, n_heads, attn_p)
        # Position-wise Feed-Forward Networks
        self.pos_ffn = PosiWiseFeedForwardNet(d_model, dff, pwffn_p)
    
    def forward(self, enc_in, enc_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        
        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_in, enc_in, enc_in, enc_attn_mask)
        
        # enc_outputs: [batch_size, src_len, d_model]
        ffn_out = self.pos_ffn(enc_outputs)
        
        return ffn_out, attn


class Encoder(nn.Module):
    def __init__(self, enc_p, pwffn_p, attn_p, n_layers, enc_model, n_heads, dff, src_len):
        """
        Args:
            enc_p: dropout ratio of Position Embeddings.
            pwffn_p: dropout ratio of PosiWiseFeedForwardNet.
            attn_p: dropout ratio of attention module.
            n_layers: number of encoder layers
            enc_model: input dimension of encoder
            n_heads: number of attention heads
            dff: dimensionf of PosiWiseFeedForwardNet
            src_len: the maximum length of all source sequences
        """
        super(Encoder, self).__init__()
        self.pos_enc = nn.Embedding.from_pretrained(positional_encode(src_len, enc_model), freeze=True)
        
        self.enc_dropout = nn.Dropout(enc_p)
        
        self.layers = nn.ModuleList([
            EncoderLayer(enc_model, n_heads, dff, pwffn_p, attn_p) for _ in range(n_layers)
        ])
    
    def forward(self, enc_inputs, mask=None):
        batch_size, src_len, d_model = enc_inputs.shape
        
        enc_pos_out = enc_inputs + self.pos_enc(torch.arange(src_len, device=enc_inputs.device))
        
        enc_output = self.enc_dropout(enc_pos_out)
        
        for layer in self.layers:
            enc_output, attn = layer(enc_pos_out, mask)
        
        return enc_output, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, pwffn_p, attn_p):
        """
        Args:
            d_model: input dimension
            n_heads: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            pwffn_p: dropout ratio of PosFFN
            attn_p: dropout ratio of attention module
        """
        assert d_model % n_heads == 0
        d_qkv = d_model // n_heads
        super(DecoderLayer, self).__init__()
        
        # MultiHeadAttention, both self-attention and encoder-decoder cross attention
        self.dec_attn = MultiHeadAttention(d_qkv, d_qkv, d_model, n_heads, attn_p)
        self.enc_dec_attn = MultiHeadAttention(d_qkv, d_qkv, d_model, n_heads, attn_p)
        
        # Position-wise Feed-Forward Networks
        self.pos_ffn = PosiWiseFeedForwardNet(d_model, dff, pwffn_p)
    
    def forward(self, enc_out, dec_in, dec_mask,  dec_enc_mask):
        # dec_self_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_self_outputs, dec_self_attn = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        
        # dec_enc_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_enc_outputs, dec_enc_attn = self.enc_dec_attn(dec_self_outputs, enc_out, enc_out, dec_enc_mask)
        
        dec_out = self.pos_ffn(dec_enc_outputs)
        
        return dec_out, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, dec_p, pwffn_p, attn_p, n_layers, dec_model, n_heads, dff, tgt_len, tgt_vocab_size):
        """
        Args:
            enc_p: dropout ratio of Position Embeddings.
            pwffn_p: dropout ratio of PosiWiseFeedForwardNet.
            attn_p: dropout ratio of attention module.
            n_layers: number of encoder layers
            dec_model: input dimension of decoder
            n_heads: number of attention heads
            dff: dimensionf of PosiWiseFeedForwardNet
            tgt_len: the maximum length of all target sequences
            tgt_vocab_size: the target vocabulary size.
        """
        super(Decoder, self).__init__()
        # output embedding
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_model)
        self.dec_dropout = nn.Dropout(p=dec_p)
        self.pos_enc = nn.Embedding.from_pretrained(positional_encode(tgt_len, dec_model), freeze=True)
        self.layers = nn.ModuleList([
            DecoderLayer(dec_model, n_heads, dff, pwffn_p, attn_p) for _ in range(n_layers)
        ])
    
    def forward(self, enc_out, dec_in, dec_mask, dec_enc_mask):
        # output embedding and position embedding
        tgt_emb = self.tgt_emb(dec_in)
        pos_emb = self.pos_enc(torch.arange(dec_in.size(1), device=dec_in.device))
        dec_out = self.dec_dropout(tgt_emb + pos_emb)
        
        # decoder layers
        for layer in self.layers:
            dec_out, dec_self_attn, dec_enc_attn = layer(enc_out, dec_out, dec_mask, dec_enc_mask)
        
        return dec_out


class Transformer(nn.Module):
    def __init__(self, frontend, encoder, decoder, dec_out_dim, vocab_size):
        super().__init__()
        self.frontend = frontend  # feature extractor
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab_size)
    
    def forward(self, enc_in, enc_in_lens, dec_in):
        batch_size = enc_in.size(0)
        device = enc_in.device
        enc_in_lens, dec_in = enc_in_lens.long(), dec_in.long()
        
        # feature extractor
        feature_out, enc_in_lens = self.frontend(enc_in, enc_in_lens)
        
        # encoder
        enc_mask = get_enc_attn_mask(batch_size, feature_out.size(1), enc_in_lens, device)
        enc_out, _ = self.encoder(feature_out, enc_mask)
        
        # decoder
        max_src_len = enc_out.size(1)
        max_tgt_len = dec_in.size(1)
        dec_mask = get_subsequent_mask(batch_size, max_tgt_len, device)
        dec_enc_mask = get_enc_dec_mask(batch_size, max_src_len, enc_in_lens, max_tgt_len, device)
        
        dec_out = self.decoder(enc_out, dec_in, dec_mask, dec_enc_mask)
        logits = self.linear(dec_out)
        
        return logits


if __name__ == "__main__":
    from feature_extractors import LinearFeatureExtractionModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('device: ', device)
    
    # constants
    batch_size = 16  # batch size
    pad_model = 512  # processed different sequence lengths in batch_size, Padded to the same length
    
    max_src_len = 100  # The maximum length of input/source sequence in the current batch_size
    feat_dim = 80  # The feature dimension of each sequence in the current batch_size

    max_tgt_len = 100  # the maximum length of output/target sequence in the current batch_size
    vocab_size = 30  # The number of candidate words for each sequence. the size of the vocabulary
    
    # dummy encoder data (batch_size, max_src_len, d)
    feat_data = torch.randn(batch_size, max_src_len, feat_dim).to(device)  # input sequence data
    feat_lens = torch.randint(1, max_src_len, (batch_size,)).to(device)  # the length of each input sequence in the batch

    # dummy decoder data (batch_size, max_tgt_len, V) / (batch_size, max_tgt_len)
    labels_data = torch.randint(0, vocab_size, (batch_size, max_tgt_len)).to(device)  # output sequence
    label_lens = torch.randint(1, max_tgt_len, (batch_size,)).to(device)  # the length of each output sequence in the batch
    
    print('input sequence：feat_data = ', feat_data.shape)
    print('the length of each input sequence in the batch：feat_lens = ', feat_lens.shape, feat_lens)
    print('output sequence：labels_data = ', labels_data.shape)
    print('the length of each output sequence in the batch：label_lens = ', label_lens.shape, label_lens,'\n')
    
    # model
    feature_extractor = LinearFeatureExtractionModel(in_dim=feat_dim, out_dim=pad_model).to(device)
    
    with torch.no_grad():
        output, feat_lens = feature_extractor(feat_data, feat_lens)
        print(f"fbank_feature: {feat_data.shape} -> {output.shape}--{type(output)}")
        print(f"feat_lens: {feat_lens.shape} ->{type(feat_lens)}", '\n')
    
    encoder = Encoder(
        enc_p=.1, pwffn_p=.1, attn_p=.1,
        n_layers=6, enc_model=pad_model, n_heads=8, dff=2048, src_len=2048
    ).to(device)
    
    decoder = Decoder(
        dec_p=.1, pwffn_p=.1, attn_p=.1,
        n_layers=6, dec_model=pad_model, n_heads=8, dff=2048, tgt_len=2048, tgt_vocab_size=vocab_size
    ).to(device)
    
    transformer = Transformer(feature_extractor, encoder, decoder, pad_model, vocab_size).to(device)
    
    # forward check
    with torch.no_grad():
        logits = transformer(feat_data, feat_lens, labels_data)
        print(f"logits: {logits.shape}")  # (batch_size, max_label_len, vocab_size)
