import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.ETSformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, Transform
import torch.nn.functional as F

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
    Paper link: https://arxiv.org/abs/2202.01381
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
            self.use_gnn = configs.use_gnn
        if configs.use_gnn:
            self.adp = adp(pred_len=configs.seq_len,device="cuda:" + configs.devices)

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in, configs.seq_len, self.pred_len, configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, self.pred_len,
                    dropout=configs.dropout,
                ) for _ in range(configs.d_layers)
            ],
        )
        self.transform = Transform(sigma=0.2)

        if self.task_name == 'classification':
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        if self.use_gnn:
            x_enc, _ = self.adp(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def anomaly_detection(self, x_enc):
        res = self.enc_embedding(x_enc, None)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def classification(self, x_enc, x_mark_enc):
        res = self.enc_embedding(x_enc, None)
        _, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growths = torch.sum(torch.stack(growths, 0), 0)[:, :self.seq_len, :]
        seasons = torch.sum(torch.stack(seasons, 0), 0)[:, :self.seq_len, :]

        enc_out = growths + seasons
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)

        # Output
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
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
