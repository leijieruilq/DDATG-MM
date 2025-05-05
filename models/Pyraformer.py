import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import Encoder
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
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs, window_size=[4,4], inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.use_gnn = configs.use_gnn
        if configs.use_gnn:
            self.adp = adp(pred_len=configs.seq_len,device="cuda:" + configs.devices)

        if self.task_name == 'short_term_forecast':
            window_size = [2,2]
        self.encoder = Encoder(configs, window_size, inner_size)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, self.pred_len * configs.enc_in)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, configs.enc_in, bias=True)
        elif self.task_name == 'classification':
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model * configs.seq_len, configs.num_class)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        return dec_out
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.encoder(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.encoder(x_enc, x_mark_enc=None)

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
            if self.use_gnn:
                x_enc, _ = self.adp(x_enc)
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
