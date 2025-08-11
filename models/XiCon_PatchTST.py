import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.AutoCon_wnorm = configs.AutoCon_wnorm
        self.d_model = configs.d_model
        # patching and embedding
        self.patch_embedding = PatchEmbedding(self.seq_len,
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 1) + 1
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

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        
        #self.head_nf = self.patch_num

        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        self.Linear = nn.Linear(configs.d_model, self.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            x_enc = (x_enc - seq_mean) / (seq_std + 1e-5)
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            x_enc = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:,-1:,:].detach()
            x_enc = (x_enc - seq_last)

        else:
            raise Exception(f'Not Supported Window Normalization:{self.AutoCon_wnorm}. Use {"{ReVIN | Mean | LastVal | Decomp}"}.')

        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        """
        B, T, C = x_enc.shape
        #x_enc = x_enc.permute(0, 2, 1).reshape(B * C, T, 1)
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) #128, 1, 336
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc) 

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        repr, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        repr = torch.reshape(
            repr, (-1, n_vars, repr.shape[-2], repr.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        repr = repr.permute(0, 1, 3, 2)

        repr = repr.reshape(B, -1, C * self.d_model)  # [bs, patch_num, nvars * d_model]

        # Decoder
        dec_out = self.head(repr.reshape(B, C, self.d_model, self.patch_num))  # z: [bs x nvars x target_window]

        dec_out = dec_out.permute(0, 2, 1)

        if self.AutoCon_wnorm == 'ReVIN':
            pred = (dec_out)*(seq_std+1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            pred = dec_out + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            pred = dec_out
        elif self.AutoCon_wnorm == 'LastVal':
            pred = dec_out + seq_last
        else:
            raise Exception()

        return pred, repr

    def get_embeddings(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        repr, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        repr = torch.reshape(
            repr, (-1, n_vars, repr.shape[-2], repr.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        repr = repr.permute(0, 1, 3, 2)
        return repr
