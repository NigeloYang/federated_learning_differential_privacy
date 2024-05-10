# -*- coding: utf-8 -*-
# Time    : 2024/5/8
# By      : Yang

import numpy as np
import torch
import torch.nn as nn


# mask
def get_len_mask(b, max_src_len, feat_len, device):
    """ encoder mask
    Args:
        b: batch-size.
        max_src_len: the max length of the all source seqeunce.
        feat_len: the feature dimension length of each source seqeunce.
        device: cuda or cpu.
    """
    # mask: [batch_size,  max_src_len,  max_src_len]
    attn_mask = torch.ones((b, max_src_len, max_src_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_len[i]] = 0
    return attn_mask.to(torch.bool)


def get_subsequent_mask(b, max_tgt_len, device):
    """ decoder mask
    Args:
        b: batch-size.
        max_tgt_len: the max length of the all target seqeunce.
        device: cuda or cpu.
    """
    #  Generate the upper triangular matrix, [batch_size, max_tgt_len, max_tgt_len]
    sub_mask = torch.triu(torch.ones((b, max_tgt_len, max_tgt_len), device=device), diagonal=1)
    return sub_mask.to(torch.bool)


def get_enc_dec_mask(b, max_feat_len, feat_lens, max_label_len, device):
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)  # (b, seq_q, seq_k)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)


# position
def positionalEncoding(seq_len, d_model):
    posi_encode = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos    # 字嵌入维度为偶数时
        posi_encode[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return posi_encode.float()  # enc_inputs: [seq_len, d_model]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)
        
        # linear projections, d_k * num_heads = d_model
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)
        
        # Normalization
        # References: <<Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification>>
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
    
    def forward(self, input_Q, input_K, input_V, attn_mask):
        print('MultiHeadAttention',  input_Q.size(), type( input_Q))
        
        batch_size = input_Q.size(0)
        len_q, len_k = input_Q.size(1), input_K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads
        
        # multi head split, d_k = depth = d_model // num_heads
        # (batch_size, seq_len, d_model) > (batch_size, seq_len_qkv, num_heads, depth) > (batch_size, num_heads, seq_len_qkv, depth)
        Q = self.W_Q(input_Q).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, num_heads, d_v).transpose(1, 2)
        
        # scaled dot product attention, pre-process mask
        # k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
        # 虽然 mask 根据其类型（填充或前瞻）有不同的形状，但是 mask 必须能进行广播转换以便求和。
        if attn_mask is not None:
            assert attn_mask.size() == (batch_size, len_q, len_k)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)  # broadcast
            attn_mask = attn_mask.bool()
        
        # calculate attention weight score:  scores =  softmax(Q*K^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        
        # attn_mask 加入到缩放的 attention weight score。
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        
        # 使用 softmax 归一化 attention weights 分数
        attention_weight = torch.softmax(scores, dim=-1)
        attention_weight = self.dropout(attention_weight)
        
        # calculate output: res = softmax(Q*K^T / sqrt(d_k)) * V
        # output.shape = (batch_size, num_heads, seq_len, depth)
        output = torch.matmul(attention_weight, V)
        
        # multi_head attention merge
        # output.shape = (batch_size, seq_len, num_heads, depth) ==> (batch_size, src_len, d_model)
        output = output.transpose(1, 2).contiguous().reshape(batch_size, -1, d_v * num_heads)
        output = self.W_out(output)
        
        return output


# Position-wise Feed-Forward Networks
class PoswiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, p=0.):
        super(PoswiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, 1, 0)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))  # (batch_size, d_model, seq_len) -> (batch_size, d_model, seq_len)
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)  # (batch_size, d_model, seq_len) -> (batch_size, d_model, seq_len)
        out = self.dropout(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_posffn, dropout_attn):
        """
        Args:
            d_model: input dimension
            num_heads: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads  # dimension of each attention head
        super(EncoderLayer, self).__init__()
        
        # MultiHeadAttention (d_k, d_v, d_model, num_heads, p=0.)
        self.multi_head_attn = MultiHeadAttention(head_dim, head_dim, d_model, num_heads, dropout_attn)
        
        # Position-wise Feed forward Neural Network
        self.poswise_ffn = PoswiseFFN(d_model, dff, p=dropout_posffn)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, enc_in, attn_mask):
        # reserve original input for later residual connections
        residual_enc = enc_in

        print('Encoder layer', enc_in.size(), type(enc_in))
        
        # MultiHeadAttention forward
        attn_output = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        
        # residual connection and norm
        norm1_out = self.norm1(residual_enc + attn_output)
        residual_out1 = norm1_out
        
        # position-wise feedforward
        ffn_out = self.poswise_ffn(norm1_out)
        
        # residual connection and norm
        norm2_out = self.norm2(residual_out1 + ffn_out)
        
        return norm2_out


class Encoder(nn.Module):
    def __init__(
        self, dropout_emb, dropout_posffn, dropout_attn, num_layers, enc_dim, num_heads, dff, src_len
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimension of PosFFN
            src_len: the maximum length of all source sequences
        """
        super(Encoder, self).__init__()
        # The maximum length of input sequence
        self.src_len = src_len
        
        # positional Encoding embedding
        self.pos_emb = nn.Embedding.from_pretrained(positionalEncoding(src_len, enc_dim), freeze=True)
        
        # add embedding dropout
        self.emb_dropout = nn.Dropout(dropout_emb)
        
        # encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )
    
    def forward(self, enc_input, mask=None):
        batch_size, src_len, d_model = enc_input.shape
        
        # add position embedding,  (batch_size, src_len, d_model)
        enc_out = enc_input + self.pos_emb(torch.arange(src_len, device=enc_input.device))
       
        
        # add embedding dropout
        enc_out = self.emb_dropout(enc_out)
        print('Encoder', enc_out.size(), type(enc_out))
        
        # encoder layers
        for layer in self.layers:
            out = layer(enc_out, mask)
        
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_posffn, dropout_attn):
        """
        Args:
            d_model: input dimension
            num_heads: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        super(DecoderLayer, self).__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads  # dimension of each attention head
        
        # MultiHeadAttention, both self-attention and encoder-decoder cross attention
        self.dec_attn = MultiHeadAttention(head_dim, head_dim, d_model, num_heads, dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(head_dim, head_dim, d_model, num_heads, dropout_attn)
        
        # Position-wise Feed-Forward Networks
        self.poswise_ffn = PoswiseFFN(d_model, dff, p=dropout_posffn)
        
        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask, cache=None, freqs_cis=None):
        # decoder's self-attention
        residual_dec = dec_in
        dec_attn_out = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        dec_norm1_out = self.norm1(residual_dec + dec_attn_out)
        
        # encoder-decoder cross attention
        residual_norm1_out = dec_norm1_out
        enc_dec_attn_out = self.enc_dec_attn(dec_norm1_out, enc_out, enc_out, dec_enc_mask)
        dec_norm2_out = self.norm2(residual_norm1_out + enc_dec_attn_out)
        
        # position-wise feed-forward networks
        residual_norm2_out = dec_norm2_out
        ffn_out = self.poswise_ffn(dec_norm2_out)
        norm3_out = self.norm3(residual_norm2_out + ffn_out)
        
        return norm3_out


class Decoder(nn.Module):
    def __init__(
        self, dropout_emb, dropout_posffn, dropout_attn,
        num_layers, dec_dim, num_heads, dff, tar_len_emb, tar_vocab_size,
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            dec_dim: input dimension of decoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tar_len_emb: the target length to be embedded.
            tar_vocab_size: the target vocabulary size.
        """
        super(Decoder, self).__init__()
        
        # output embedding
        self.out_emb = nn.Embedding(tar_vocab_size, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)  # embedding dropout
        
        # position embedding
        self.pos_emb = nn.Embedding.from_pretrained(positionalEncoding(tar_len_emb, dec_dim), freeze=True)
        
        # decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )
    
    def forward(self, labels, enc_out, dec_mask, dec_enc_mask, cache=None):
        # output embedding and position embedding
        out_emb = self.out_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(out_emb + pos_emb)
        
        # decoder layers
        for layer in self.layers:
            dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        
        return dec_out


class Transformer(nn.Module):
    def __init__(self, feat_frontend, encoder, decoder, dec_out_dim, vocab):
        super().__init__()
        self.feat_frontend = feat_frontend  # feature extractor
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)
    
    def forward(self, fbank_feature, feat_lens, labels):
        feat_lens, labels = feat_lens.long(), labels.long()
        b = fbank_feature.size(0)
        device = fbank_feature.device
        
        # frontend
        feature_out = self.feat_frontend(fbank_feature)
        print('feature_extractor(fbank_feature): ', feature_out.shape)
        max_feat_len = feature_out.size(1)  # compute after frontend because of optional subsampling
        max_label_len = labels.size(1)
        
        # encoder
        enc_mask = get_len_mask(b, max_feat_len, feat_lens, device)
        enc_out = self.encoder(feature_out, enc_mask)
        
        # decoder
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, feat_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        
        logits = self.linear(dec_out)
        
        return logits


if __name__ == "__main__":
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', device)
    
    # constants
    batch_size = 16  # batch size
    max_feat_len = 100  # the maximum length of input feature sequence
    max_label_len = 50  # the maximum length of output labels sequence
    fbank_dim = 80  # the dimension of input feature
    hidden_dim = 512  # the dimension of forward hidden layer
    vocab_size = 26  # the size of vocabulary
    
    # dummy data
    fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim).to(device)  # input sequence
    feat_lens = torch.randint(1, max_feat_len, (batch_size,)).to(device)  # the length of each input sequence in the batch
    labels = torch.randint(0, vocab_size, (batch_size, max_label_len)).to(device)  # output sequence
    label_lens = torch.randint(1, max_label_len, (batch_size,)).to(device)  # the length of each output sequence in the batch
    print('input sequence：fbank_feature = ', fbank_feature.shape)
    print('the length of each input sequence in the batch：feat_lens = ', feat_lens.shape)
    print('output sequence：labels = ', labels.shape)
    print('the length of each output sequence in the batch：label_lens = ', label_lens.shape)
    
    # create model.
    # a linear layer to simulate the audio feature extractor
    feature_extractor = nn.Linear(fbank_dim, hidden_dim).to(device)
    
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, enc_dim=hidden_dim, num_heads=8, dff=2048, src_len=2048
    ).to(device)
    
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, dec_dim=hidden_dim, num_heads=8, dff=2048, tar_len_emb=2048, tar_vocab_size=vocab_size
    ).to(device)
    
    transformer = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size)
    transformer.to(device)
    
    # forward check
    logits = transformer(fbank_feature, feat_lens, labels)
    print(f"logits res: {logits.shape}")  # (batch_size, max_label_len, vocab_size)
