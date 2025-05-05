import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding

class adp(nn.Module):
    def __init__(self, pred_len, device, emb_dim=8, dropout=0.5):
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
        return torch.einsum("btm,btl->bml",w_x,x) + x, x_p

class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.use_gnn = configs.use_gnn
        if configs.use_gnn:
            self.adp = adp(pred_len=configs.seq_len + self.pred_len,device="cuda:" + configs.devices)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        else:
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # add placeholder
        if self.use_gnn:
            x_enc, _ = self.adp(x_enc)
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        return enc_out  # [B, L, D]

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        return enc_out  # [B, L, D]

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
