import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import DataEmbedding
from layers.dilated_conv import DilatedConvEncoder
from layers.RevIN import RevIN


class PatchMixerLayer(nn.Module):
    def __init__(self,dim,a,kernel_size = 8):
        super().__init__()
        self.Resnet =  nn.Sequential(
            nn.Conv1d(dim,dim,kernel_size=kernel_size,groups=dim,padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim,a,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )
    def forward(self,x):
        x = x +self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                   # x: [batch * n_val, a, d_model]
        return x

class PatchMixer_backbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)
    def forward(self, x):
        x,u = self.model(x)
        return x,u
class Backbone(nn.Module):
    def __init__(self, configs,):
        super().__init__()

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size
        self.hidden_dims = configs.d_model
        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1
        # if configs.a < 1 or configs.a > self.patch_num:
        #     configs.a = self.patch_num
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.depth = configs.e_layers
        self.repr_dims =  int(self.hidden_dims * self.patch_num * self.hidden_dims / self.lookback)
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(dim=self.patch_num, a=self.patch_num, kernel_size=self.kernel_size))
        self.W_P = nn.Linear(self.patch_size, self.d_model)  

        self.dropout = nn.Dropout(self.dropout)

        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.Linear = nn.Linear(self.d_model, self.lookback)

    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]

        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]  

        x = self.W_P(x)                                                              # x: [batch, n_val, patch_num, hidden_dims]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))      # x: [batch * n_val, patch_num, hidden_dims]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            repr = PatchMixer_block(x)
            #repr = self.Linear(repr)

        #repr = repr.permute(0,2,1)    
        #x = self.head1(x)
        #x = u + x
        #x = torch.reshape(x, (bs , nvars, -1)) # x: [batch, pred_len, nvars]
                                              
        #x = x.permute(0, 2, 1)

        #repr = repr.permute(0,2,1)

        return  repr, u

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs, affine = True, subtract_last = False ):
        super(Model, self).__init__()
        self.nvals = configs.enc_in
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.lookback = configs.seq_len
        self.stride = configs.stride
        self.forecasting = configs.pred_len
        
        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_dims = configs.d_model
        self.d_model = configs.d_model
        self.head_dropout = configs.head_dropout
        self.depth = configs.e_layers
        self.patch_size = configs.patch_len
        self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1
        
        self.a = self.patch_num

        #self.repr_dims = configs.d_ff
        self.repr_dims =  int(self.hidden_dims * self.patch_num * self.hidden_dims / self.seq_len)
        self.XiCon = configs.XiCon

        self.AutoCon_wnorm = configs.AutoCon_wnorm
        self.AutoCon_multiscales = configs.AutoCon_multiscales

        enc_in = 1  # Channel Independence

        self.enc_embedding = DataEmbedding(enc_in, self.hidden_dims, configs.embed, configs.freq, dropout=configs.dropout)
        self.feature_extractor = PatchMixer_backbone(configs)

        self.repr_dropout = nn.Dropout(p=0.1)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)

        self.input_decom = series_decomp(25)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        """
        self.head1 =  nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.repr_dims * self.lookback, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        """
        self.head1 =  nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.hidden_dims, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        #self.revin = revin
        
        if self.AutoCon_wnorm == 'ReVIN': self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        # x: [Batch, Input length, Channel]

        if self.AutoCon_wnorm == 'ReVIN':

            
            long_x = self.revin_layer(x_enc, 'norm')

            """
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
            """

        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:,-1:,:].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception(f'Not Supported Window Normalization:{self.AutoCon_wnorm}. Use {"{ReVIN | Mean | LastVal | Decomp}"}.')

        B, T, C = long_x.shape
        #long_x = long_x.permute(0, 2, 1).reshape(B * C, T, 1)
        #long_x = long_x.permute(0, 2, 1)


        repr, u= self.feature_extractor(long_x)
        #repr = repr.permute(0,2,1)

        #x = self.head1(repr.permute(0,2,1))
        x = self.head1(repr)
        x = u + x
        
        """
        x = torch.reshape(x, (B*C , 1, -1)) # x: [batch * nvars,1 , pred_len]
        x = x.permute(0, 2, 1)

        """

        trend_outs = x.reshape(B, C, self.pred_len).permute(0, 2, 1)


        if self.AutoCon_wnorm == 'ReVIN':
            pred = self.revin_layer(trend_outs, 'denorm')
            #pred = (trend_outs)*(seq_std+1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            pred = trend_outs + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            pred = trend_outs
        elif self.AutoCon_wnorm == 'LastVal':
            pred = trend_outs + seq_last
        else:
            raise Exception()

        if self.XiCon:
            return pred, repr  # [Batch, Output length, Channel]
        else:
            return pred


    def get_embedddings(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):

        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:,-1:,:].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception()

        if self.ablation != 2:
            B, T, C = long_x.shape
            long_x = long_x.permute(0, 2, 1).reshape(B * C, T, 1)
            long_x = long_x.permute(0, 2, 1)
            long_x = long_x.permute(0,2,1)
            repr = self.repr_dropout(self.feature_extractor1(long_x)).transpose(1, 2) # B x Co x T
        else:
            repr = None

        return repr



