import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class adp(nn.Module):
    def __init__(self, pred_len, device, emb_dim=8, dropout=0.3):
        super().__init__()
        self.c_emb = nn.Parameter(torch.randn(pred_len, emb_dim))
        self.adj_proj1 = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.adj_proj2 = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.decouple_m_real = nn.Parameter(torch.randn(int(pred_len//2) + 1, int(pred_len//2) + 1))
        self.decouple_m_imag = nn.Parameter(torch.randn(int(pred_len//2) + 1, int(pred_len//2) + 1))
        nn.init.xavier_normal_(self.c_emb)
        nn.init.xavier_normal_(self.adj_proj1)
        nn.init.xavier_normal_(self.adj_proj2)
        nn.init.xavier_normal_(self.decouple_m_real)
        nn.init.xavier_normal_(self.decouple_m_imag)
        self.drop = nn.Dropout(p=dropout)
        self.neg_inf = -1e9 * torch.eye(pred_len, device=device)
        self.real_mlp = nn.Linear(int(pred_len//2) + 1, pred_len)
        self.imag_mlp = nn.Linear(int(pred_len//2) + 1, pred_len)
    def forward(self,x):
        c_emb1 = torch.mm(self.c_emb, self.adj_proj1)
        c_emb2 = torch.mm(self.c_emb, self.adj_proj2)
        adj = torch.matmul(c_emb1, c_emb2.transpose(0,1))
        adj = F.relu(adj)
        adj = adj + self.neg_inf
        adj = torch.where(adj<=0.0,self.neg_inf,adj)
        adj = F.softmax(adj, dim=-1)
        w_x = F.gelu(self.drop(torch.einsum("btl,tm->btm",[x,adj]))) #依赖关系
        x_fft = torch.fft.rfft(x,dim=-2)
        real_part = F.tanh(torch.einsum("ml,bme->ble", self.decouple_m_real, x_fft.real)) * x_fft.real
        imag_part = F.tanh(torch.einsum("ml,bme->ble", self.decouple_m_imag, x_fft.imag)) * x_fft.imag
        x_fft = torch.complex(real_part, imag_part)
        x_p = torch.fft.irfft(x_fft,dim=-2)
        x = x - x_p
        return torch.einsum("btm,btl->bml",w_x,x) + x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_gnn = configs.use_gnn
        if configs.use_gnn:
            self.adp = adp(pred_len=configs.d_model,device="cuda:" + configs.devices)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self.use_gnn:
            #enc_out = enc_out
            enc_out = self.adp(enc_out.permute(0,2,1))
            enc_out = enc_out.permute(0,2,1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
