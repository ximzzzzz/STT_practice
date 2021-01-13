import torch
import torch.nn as nn
import conformer
import easydict
import numpy as np
import pandas as pd
import json
import os
import librosa
import torch.nn.functional as F

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
#         print(f'x shape : {x.shape}')
        t, n = x.size(0), x.size(1)
#         x = x.contiguous().view(t * n, -1)
        x = x.reshape((t * n, -1))
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
#         print(f'BatchRNN shape : {x.shape}')
        total_length = x.size(0)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        self.flatten_parameters()
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length = total_length)
#         x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x

class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
#             pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, x, x_mask=None):
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
#         x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
#         x = self.out(x.transpose(1,2).reshape((b, t, c*f)))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
    
    
# class Enc_pre(torch.nn.Module):
#     def __init__(self, args):
#         super(Enc_pre, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 256, kernel_size=3, stride=2)
#         self.conv2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=2)
#         self.linear = torch.nn.Linear(256, args.dim)
#         self.dropout = torch.nn.Dropout(p=0.1)


#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.permute(0,3,2,1) ######## unsure yet
#         x = self.linear(x)
#         x = self.dropout(x)
        
#         return x
    
class Enc_body(torch.nn.Module):
    def __init__(self, args):
        super(Enc_body, self).__init__()
        
        conformer_blocks = []
        for _ in range(args.n_enc_layers):
            conformer_blocks.append(
                conformer.ConformerBlock(       
                    dim = args.dim,
                    dim_head = args.dim_head,
                    heads = args.heads ,
                    ff_mult = args.ff_mult,
                    conv_expansion_factor = args.conv_expansion_factor,
                    conv_kernel_size = args.conv_kernel_size,
                    attn_dropout = args.attn_dropout,
                    ff_dropout = args.ff_dropout,
                    conv_dropout = args.conv_dropout
                )
            )
        self.conformer_layers = torch.nn.ModuleList(conformer_blocks)
        
    def forward(self, x):
        for layer in self.conformer_layers:
            x = layer(x)
        
        return x
    
class Conformer(torch.nn.Module):
    def __init__(self, args):
        super(Conformer, self).__init__()
#         self.enc_pre = Enc_pre(args)
        self.enc_pre = Conv2dSubsampling(80, 256, 0.1)
        self.enc_body = Enc_body(args)
#         self.dec_body = torch.nn.LSTM(args.dim, args.dec_dim, num_layers=1 )
        self.dec_body = BatchRNN(input_size = args.dim, 
                                 hidden_size = args.dec_dim,
#                                  hidden_size = args.n_classes,
                                 bidirectional=False,
                                 batch_norm=False
                                )
        self.fc = torch.nn.Linear(args.dec_dim, args.n_classes)
#         self.fc = torch.nn.Linear(args.dim, args.n_classes)
#         self.model = torch.nn.ModuleList(self.enc_pre + self.enc_body + self.dec_body + self.fc)
    
    def forward(self, x, input_percentages) :
        x = x.permute(0, 1, 3, 2) # N, C, F, T -> N, C, T, F 
        x,_ = self.enc_pre(x)
        x = self.enc_body(x)
        x = x.transpose(0,1) # T, N ,H
        input_sizes = input_percentages.mul_(x.size(0)).int()
        x = self.dec_body(x,input_sizes)
        x = self.fc(x)
        x = x.transpose(0,1)
#         x = torch.softmax(x, dim=-1)
        x = F.log_softmax(x, dim=-1)
    
        return x, input_sizes