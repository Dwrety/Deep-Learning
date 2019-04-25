import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import numpy as np



class LockedDropout(nn.Module):
    def __init__(self, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

    def forward(self, x, dropout=0.5):
        if dropout == 0 or not self.training:
            return x 
        if not self.batch_first:
            x = x.permute(1, 0, 2).contiguous()
        mask = x.data.new(x.size(0), 1, x.size(2))
        mask = mask.bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        if not self.batch_first:
            return (mask*x).permute(1, 0, 2).contiguous()
        return mask * x


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    X = nn.functional.embedding(words, masked_embed_weight,
    embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
    )
    return X


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        self._setweights()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = nn.Parameter(torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training))
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, sample_rate=2, sample_type='concat', num_layers=1,
                 rnn_cell='LSTM', dropout_rate=0.0, wdrop=0, bidirectional=True, batch_first=True):
        super(pBLSTM, self).__init__()
        '''
        params:
            input_dim: int, input dimension of the LSTM layer.
            hidden_dim: int, number of hidden units.
            sample_rate: int, down sampling argument
            sample_type: sample method
            num_layers: default 1
            rnn_cell: default LSTM
            dropout_rate: dropout rate of input tensor
            bidirectional: bool, if True, perform bidirectional LSTM
        '''
        self.sample_type = sample_type
        self.batch_first = batch_first
        self.sample_rate = sample_rate
        self.lockdrop = LockedDropout()
        self.rnn_layer = getattr(nn, rnn_cell.upper())(input_dim*2, hidden_dim>>1, 
                                                    bidirectional=bidirectional, 
                                                    num_layers=num_layers,
                                                    dropout=dropout_rate, 
                                                    batch_first=batch_first)
        if wdrop > 0:
            self.rnn_layer = WeightDrop(self.rnn_layer, ['weight_hh_l0', 'weight_hh_l0_reverse'], dropout=wdrop)

    def forward(self, input_x, hidden=None, state_len=None, pack_input=False):
        '''
        The input needs to have an even number of seq_len, 
        which means the last time batch is deleted.
        params:
            input_x: A PyTorch PackedSequence object
                     or a 3D numpy array of size [batch_size x seq_len(sorted descend) x input_dim]
            hidden: A tuple of the initial hidden state of LSTM cell; (init_h, init_c)
            state_len: A length vector of corresponding to dim=1 of input_x
            pack_input: bool; if True, perform a pack_padded_sequence before forward method. 
        '''
        # print(state_len)
        if self.sample_rate > 1:
            batch_size, max_state_len, feature_dim = input_x.shape

        if self.sample_type == 'drop':
            input_x = input_x[:,::self.sample_rate,:]
            state_len = (state_len.float()/self.sample_rate).ceil_().int()

        elif self.sample_type == 'concat':
            if max_state_len % self.sample_rate != 0:
                input_x = input_x[:,:-(max_state_len%self.sample_rate), :]
            state_len = state_len - state_len % self.sample_rate
            state_len = (state_len.float()/self.sample_rate).ceil_().int()
            input_x = input_x.contiguous().view(batch_size, int(max_state_len/self.sample_rate), feature_dim*self.sample_rate)
        else:
            raise ValueError("Unsupport Sample Type" + self.sample_type)

        if pack_input:
            assert state_len is not None, "Please specify sequence length for pack_padded_sequence."
            # requires a sorted batch
            input_x = rnn_utils.pack_padded_sequence(input_x, state_len, batch_first=self.batch_first)

        output, hidden = self.rnn_layer(input_x, hidden)

        if pack_input:
            output, state_len = rnn_utils.pad_packed_sequence(output, batch_first=self.batch_first)

        if self.training:
            output = self.lockdrop(output, dropout=0.2)

        # if self.sample_rate > 1:
        #     batch_size, seq_len, feature_dim = output.shape
        #     if self.sample_type == 'drop':
        #         output = output[:,::self.sample_rate,:]
        #         state_len = (state_len.float()/self.sample_rate).ceil_().int()
        #     elif self.sample_type == 'concat':
        #         if seq_len % self.sample_rate != 0:
        #             output = output[:,:-(seq_len%self.sample_rate), :]
        #         state_len = state_len - state_len % self.sample_rate
        #         state_len = (state_len.float()/self.sample_rate).ceil_().int()
        #         output = output.contiguous().view(batch_size, int(seq_len/self.sample_rate), feature_dim*self.sample_rate)
        #     else:
        #         raise ValueError("Unsupport Sample Type" + self.sample_type)
        if state_len is not None:
            return output, hidden, state_len
        return output, hidden


class ConvBlock(nn.Module):
    def __init__(self, input_channels=40, output_channels=512, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.lockdrop = LockedDropout()
        self.convlayer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels>>2, kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm1d(num_features=output_channels>>2),
            nn.ELU())
        self.convlayer2 = nn.Sequential(
            nn.Conv1d(in_channels=output_channels>>2, out_channels=output_channels>>1, kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm1d(num_features=output_channels>>1),
            nn.ELU())
        self.convlayer3 = nn.Sequential(
            nn.Conv1d(in_channels=output_channels>>1, out_channels=output_channels, kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm1d(num_features=output_channels),
            nn.ELU())

    def forward(self, input_x, seq_len):
        '''
        params:
            input_x: a Torch FloatTensor size [batch_size x max_len x 40]
            seq_len: a Torch IntTensor size [batch_size, ]
        '''
        input_x = input_x.permute(0,2,1)
        output = self.convlayer1(input_x)
        # output = self.lockdrop(output, dropout=self.dropout_rate)
        seq_len = (seq_len-3)//2 + 1
        output = self.convlayer2(output)
        # output = self.lockdrop(output, dropout=self.dropout_rate)
        seq_len = (seq_len-3)//2 + 1
        output = self.convlayer3(output)
        # output = self.lockdrop(output, dropout=self.dropout_rate)
        seq_len = (seq_len-3)//2 + 1
        output = output.permute(0,2,1).contiguous()
        if self.training:
            output = self.lockdrop(output, dropout=self.dropout_rate)
        return output, seq_len


class Listener(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=512, dropout_rate=0.0, wdrop=0.0, lockdropout=0.2, num_layers=3, encoder_type='pblstm'):
        super(Listener, self).__init__()
        self.dropout_rate = dropout_rate
        self.encoder_type = encoder_type
        self.wdrop = wdrop
        
        self.lockdropout = lockdropout
        self.lockdrop = LockedDropout()
        if self.encoder_type == 'pblstm':
            self.encoder = nn.ModuleList([nn.LSTM(input_dim, hidden_dim>>1, bidirectional=True, batch_first=True),
                                          pBLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate, wdrop=wdrop, batch_first=True),
                                          pBLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate, wdrop=wdrop, batch_first=True),
                                          pBLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate, wdrop=wdrop, batch_first=True)])

        elif self.encoder_type == 'cnn':
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.encoder = nn.ModuleList([ConvBlock(input_channels=40, output_channels=hidden_dim, dropout_rate=0.2)] + 
                [WeightDrop(nn.LSTM(hidden_dim, hidden_dim>>1, num_layers=1, bidirectional=True, dropout=0, batch_first=True), ['weight_hh_l0', 'weight_hh_l0_reverse'], dropout=wdrop) \
                 for l in range(num_layers)])
        else: raise ValueError("Unsupport Sample Type" + self.encoder_type)

    def forward(self, input_x, state_len):
        '''
        params:
            input_x: FloatTensor [batch_size x max_seq_len x 40]
            state_len: IntTensor [batch_size, ]
        '''
        # print(state_len)
        if self.encoder_type == 'pblstm':
            output = input_x
            for l, layer in enumerate(self.encoder):
                if l == 0:
                    output, hidden = layer(output)
                else:

                    output, hidden, state_len = layer(output, state_len=state_len, pack_input=True)

        elif self.encoder_type == 'cnn':
            input_x = input_x.permute(0, 2, 1)
            output = self.bn1(input_x).permute(0, 2, 1).contiguous()
            for l, layer in enumerate(self.encoder):
                if l == 0:
                    output, state_len = layer(output, state_len)
                else:
                    output, hidden = layer(output)
                    if self.training:
                        output = self.lockdrop(output, dropout=self.lockdropout)
        # print(output.shape)                        
        return output, state_len         


# class pBLSTMLayer(nn.Module):
#     def __init__(self, input_feature_dim, hidden_dim, rnn_unit='LSTM', dropout_rate=0.0):
#         super(pBLSTMLayer, self).__init__()
#         self.rnn_unit = getattr(nn, rnn_unit.upper())
#         # feature dimension will be doubled since time resolution reduction
#         self.BLSTM = self.rnn_unit(input_feature_dim * 2, hidden_dim, 1, bidirectional=True,
#                                    dropout=dropout_rate, batch_first=True)

#     # BLSTM layer for pBLSTM
#     # Step 1. Reduce time resolution to half
#     # Step 2. Run through BLSTM
#     def forward(self, input_x):
#         batch_size = input_x.size(0)
#         timestep = input_x.size(1)
#         # make input len even number
#         if timestep % 2 != 0:
#             input_x = input_x[:, :-1, :]
#             timestep -= 1
#         feature_dim = input_x.size(2)
#         # Reduce time resolution
#         input_x = input_x.contiguous().view(batch_size, timestep // 2, feature_dim * 2)
#         # Bidirectional RNN
#         output, hidden = self.BLSTM(input_x)
#         return output, hidden


# class Listener2(nn.Module):
#     def __init__(self, input_feature_dim=40, listener_hidden_dim=256, dropout_rate=0.0, dropout=0, dropouth=0, dropouti=0):
#         super(Listener2, self).__init__()
#         self.cnns = torch.nn.Sequential(
#             nn.BatchNorm1d(input_feature_dim),  # normalize input channels
#             #             nn.Conv1d(input_feature_dim, nhid, 3, padding=1),
#             #             nn.BatchNorm1d(nhid),
#             #             nn.Hardtanh(inplace=True),
#         )
#         # Listener RNN layer
#         self.rnn1 = pBLSTMLayer(input_feature_dim, listener_hidden_dim,
#                                 dropout_rate=dropout_rate)
#         self.rnn2 = pBLSTMLayer(listener_hidden_dim * 2, listener_hidden_dim,
#                                 dropout_rate=dropout_rate)
#         self.rnn3 = pBLSTMLayer(listener_hidden_dim * 2, listener_hidden_dim,
#                                 dropout_rate=dropout_rate)
#         self.lockdrop = LockedDropout()
#         self.dropouti = dropouti
#         self.dropouth = dropouth
#         self.dropout = dropout

#     def forward(self, frames, seq_sizes):
#         # frames: (max_seq_len, batch_size, channels)

#         frames = frames.permute(0, 2, 1).contiguous()
#         # frames: (batch, channels, seq_len)
#         frames = self.cnns(frames)

#         output = frames.permute(0, 2, 1)
#         # output: (batch_size, max_seq_len, channels)

#         output = self.lockdrop(output, self.dropouti)
#         output, _ = self.rnn1(output)
#         output = self.lockdrop(output, self.dropouth)
#         output, _ = self.rnn2(output)
#         output = self.lockdrop(output, self.dropouth)
#         output, _ = self.rnn3(output)
#         output = self.lockdrop(output, self.dropout)
#         print(output.shape)

#         # shorten for 8x
#         out_seq_sizes = [size // 8 for size in seq_sizes]

#         return output, out_seq_sizes


class Attention(nn.Module):
    # attention (Luong et al 2015)
    def __init__(self, encode_dim, decode_dim, context_dim=256, attention_mode='global', method='general', num_heads=1, **kwargs):
        super(Attention, self).__init__()
        # **kwargs: {mlp_dim: 512, }

        self.attention_mode = attention_mode.lower()
        self.method = method.lower()
        self.hidden = None
        self.context_dim = context_dim
        self.softmax = nn.Softmax(dim=-1)
        self.Wc = nn.Linear(encode_dim, context_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if num_heads > 1:
            # TODO
            pass
        if self.attention_mode == 'local':
            # TODO implenmentation of "local-p" method
            self.window = 40
            self.Wp = nn.Linear(decode_dim, self.context_dim, bias=False)
            self.Vp = nn.Linear(self.context_dim, 1, bias=False)

        if method == 'dot':
            assert encode_dim == decode_dim, "encoder dimension {} does not match decoder dimension {}, use general instead of dot.".format(encode_dim, decode_dim)
        elif self.method == 'general':
            self.Wa = nn.Linear(encode_dim, decode_dim)
        elif self.method == 'concat':
            pass # TODO
        elif self.method == 'mlp':
            num_units = kwargs['mlp_dim']
            self.mlp = nn.Sequential(nn.Linear(encode_dim, num_units),
                                    self.act,
                                    nn.Linear(num_units, num_units),
                                    self.act,
                                    nn.Linear(num_units, decode_dim),
                                    self.act)
        else: raise ValueError("Unsupport attention kernel " + method)
    
    def _window(self, hs, ht, state_len):
        assert self.window <= 0.5 * state_len.min(), "Please pick a smaller window size!"
        S = state_len.clone()
        pts = S * F.sigmoid(self.Vp(F.tanh(self.Wp(ht)))).unsqueeze(dim=1)
        # [batch_size, 1]
        std = 0.25 * self.window
        hs_w = Variable(torch.zeros((hs.size(0), 1+2*self.window, hs.size(2))))
        Gauss = Variable(torch.zeros((hs.size(0), 1+2*self.window))) # [batch_size, 2D+1]
        for b in range(ht.size(0)):
            leng = state_len[b]            
            p = int(S[b].item())
            p = max(self.window, p)
            p = min(leng-self.window-1, p)
            hs_w[b] = hs[b, p-self.window:p+self.window+1]
            Gauss[b] = torch.Tensor([torch.exp(-(j-p).pow(2)/(2*std**2)) for j in range(p-self.window, p+self.window+1)]) # len: p1 - p0
        new_len = torch.Tensor(state_len.shape).fill_(1+2*self.window).int()
        return hs, new_len, Gauss

    def score(self, hs, ht, state_len):
        '''
        params:
            hs: source hidden state. FloatTensor [batch_size x max_state_len x encode_dim]
            ht: target hidden state. FloatTensor [batch_size x decode_dim]
        '''
        if self.attention_mode == 'local':
            hs, state_len, Gauss = self._window(hs, ht, state_len)

        if self.method == 'dot':
            energy = self.act(ht.unsqueeze(1).bmm(hs.transpose(1,2)).squeeze(dim=1))
            # e: [batch_size, (1)squeezed , max_state_len]
        elif self.method == 'general':
            # Wa(hs): [batch_size, max_state_len, decode_dim] 
            energy = self.act(ht.unsqueeze(1).bmm(self.Wa(hs).transpose(1,2)).squeeze(dim=1))
            # e: [batch_size, (1)squeezed , max_state_len]
        elif self.method == 'concat': pass # TODO
        elif self.method == 'mlp': 
            energy = ht.unsqueeze(1).bmm(self.mlp(hs).transpose(1,2)).squeeze(dim=1)
        mask = Variable(energy.data.new(energy.size(0), energy.size(1)).fill_(1), requires_grad=False)
        for i, length in enumerate(state_len):
            mask[i, :length] = 0
        mask = mask.byte()
        energy = energy.masked_fill(mask, -10000)
        alpha = self.softmax(energy)
        if self.attention_mode == 'local':
            alpha = alpha * Gauss
        # alpha = alpha.unsqueeze(1)
        # alpha: [batch_size, 1, state_len]
        return alpha, hs

    def forward(self, hs, ht, state_len):
        '''
        params:
            hs: source hidden state. FloatTensor [batch_size x state_len x encode_dim]
            ht: target hidden state. FloatTensor [batch_size x 1 x decode_dim]
        '''
        alpha, hs = self.score(hs, ht, state_len)
        # [batch_size, 1, state_len]
        context = alpha.unsqueeze(1) @ self.Wc(hs)
        # [batch_size, 1, context_dim]
        return alpha, context.squeeze(1)


class AttentionQK(nn.Module):
    def __init__(self, encode_dim, decode_dim, query_dim=128, key_dim=128 , context_dim=256, attention_mode='global', method='dot', num_heads=1, **kwargs):
        super(AttentionQK, self).__init__()

        self.attention_mode = attention_mode.lower()
        self.method = method.lower()
        self.hidden = None
        self.context_dim = context_dim
        self.softmax = nn.Softmax(dim=-1)

        self.query_fc = nn.Linear(decode_dim, query_dim)
        self.key_fc = nn.Linear(encode_dim, key_dim)
        self.Wc = nn.Sequential(nn.Linear(encode_dim, context_dim), nn.LeakyReLU(negative_slope=0.3, inplace=True))
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        if num_heads > 1:
            # TODO
            pass
        if self.attention_mode == 'local':
            # TODO implenmentation of "local-p" method
            self.window = 40
            self.Wp = nn.Linear(decode_dim, self.context_dim, bias=False)
            self.Vp = nn.Linear(self.context_dim, 1, bias=False)

        if method == 'dot':
            assert query_dim == key_dim, "query dimension {} does not match key dimension {}, use general instead of dot.".format(query_dim, key_dim)

        elif self.method == 'general':
            self.Wa = nn.Linear(key_dim, query_dim)

        elif self.method == 'concat':
            pass # TODO

        elif self.method == 'mlp':
            num_units = kwargs['mlp_dim'] if kwargs else 128
            self.mlp = nn.Sequential(nn.Linear(key_dim, num_units),
                                    self.act,
                                    nn.Linear(num_units, num_units),
                                    self.act,
                                    nn.Linear(num_units, query_dim),
                                    self.act)

        else: raise ValueError("Unsupport attention kernel " + method)
    
    def _window(self, hs, ht, state_len):
        assert self.window <= 0.5 * state_len.min(), "Please pick a smaller window size!"
        S = state_len.clone()
        pts = S * F.sigmoid(self.Vp(F.tanh(self.Wp(ht)))).unsqueeze(dim=1)
        # [batch_size, 1]
        std = 0.25 * self.window
        hs_w = Variable(torch.zeros((hs.size(0), 1+2*self.window, hs.size(2))))
        Gauss = Variable(torch.zeros((hs.size(0), 1+2*self.window))) # [batch_size, 2D+1]
        for b in range(ht.size(0)):
            leng = state_len[b]            
            p = int(S[b].item())
            p = max(self.window, p)
            p = min(leng-self.window-1, p)
            hs_w[b] = hs[b, p-self.window:p+self.window+1]
            Gauss[b] = torch.Tensor([torch.exp(-(j-p).pow(2)/(2*std**2)) for j in range(p-self.window, p+self.window+1)]) # len: p1 - p0
        new_len = torch.Tensor(state_len.shape).fill_(1+2*self.window).int()
        return hs, new_len, Gauss

    def score(self, hs, ht, state_len):
        '''
        params:
            hs: source hidden state. FloatTensor [batch_size x max_state_len x encode_dim]
            ht: target hidden state. FloatTensor [batch_size x decode_dim]
        '''
        if self.attention_mode == 'local':
            hs, state_len, Gauss = self._window(hs, ht, state_len)

        if self.method == 'dot':
            query = self.act(self.query_fc(ht))
            key = self.act(self.key_fc(hs))
            energy = query.unsqueeze(1).bmm(key.transpose(1,2)).squeeze(dim=1)
            # e: [batch_size, (1)squeezed , max_state_len]

        elif self.method == 'general':
            query = self.act(self.query_fc(ht))
            key = self.act(self.key_fc(hs))
            # Wa(hs): [batch_size, max_state_len, decode_dim] 
            energy = query.unsqueeze(1).bmm(self.Wa(key).transpose(1,2)).squeeze(dim=1)
            # e: [batch_size, (1)squeezed , max_state_len]

        elif self.method == 'concat': pass # TODO
        elif self.method == 'mlp':
            query = self.act(self.query_fc(ht))
            key = self.act(self.key_fc(hs))
            energy = query.unsqueeze(1).bmm(self.mlp(key).transpose(1,2)).squeeze(dim=1)
        # mask = Variable(energy.data.new(energy.size(0), energy.size(1)).fill_(1), requires_grad=False)
        # for i, length in enumerate(state_len):
        #     mask[i, :length] = 0
        # mask = mask.byte()    
        # energy = energy.masked_fill(mask, -10000)
        # alpha = self.softmax(energy)
        mask = Variable(energy.data.new(energy.size(0), energy.size(1)).zero_(), requires_grad=False)
        for i, length in enumerate(state_len):
            mask[i, :length] = 1

        alpha = self.softmax(energy)
        alpha = alpha * mask
        alpha = alpha / torch.sum(alpha, dim=1).unsqueeze(1).expand_as(alpha)

        if self.attention_mode == 'local':
            alpha = alpha * Gauss
        return alpha, hs

    def forward(self, hs, ht, state_len):
        '''
        params:
            hs: source hidden state. FloatTensor [batch_size x state_len x encode_dim]
            ht: target hidden state. FloatTensor [batch_size x 1 x decode_dim]
        '''
        alpha, hs = self.score(hs, ht, state_len)
        # [batch_size, 1, state_len]
        context = alpha.unsqueeze(1) @ self.Wc(hs)
        # [batch_size, 1, context_dim]
        return alpha, context.squeeze(1)


# class Attention3(nn.Module):
#     def __init__(self, key_query_dim=128, speller_query_dim=256, listener_feature_dim=512, context_dim=128):
#         # context_dim: C, key_query_dim: C'
#         super(Attention3, self).__init__()
#         self.softmax = nn.Softmax(dim=-1)
#         self.fc_query = nn.Linear(speller_query_dim, key_query_dim)
#         self.fc_key = nn.Linear(listener_feature_dim, key_query_dim)
#         self.fc_value = nn.Linear(listener_feature_dim, context_dim)
#         self.activate = torch.nn.LeakyReLU(negative_slope=0.2)

#     def forward(self, decoder_state, listener_feature, seq_sizes):
#         # listener_feature: B, L, C
#         # print("listener_feature.size()", listener_feature.size())
#         # if not self.training:
#         #     print(listener_feature.shape)
#         query = self.activate(self.fc_query(decoder_state))
#         # key = self.activate(SequenceWise(self.fc_key, listener_feature))
#         # print(key.shape)
#         key = self.activate(self.fc_key(listener_feature))
#         # print(key1.shape)


#         # query: B, 1, C'
#         # key  : B, L, C'

#         # print("query.size()", query.size())
#         # print("key.size()", key.size())
#         energy = torch.bmm(query.unsqueeze(1), key.transpose(1, 2)).squeeze(dim=1)

#         # energy/attention_score: B, L
#         # masked softmax
#         mask = Variable(energy.data.new(energy.size(0), energy.size(1)).zero_(), requires_grad=False)
#         for i, size in enumerate(seq_sizes):
#             mask[i, :size] = 1
#         attention_score = self.softmax(energy)
#         attention_score = mask * attention_score
#         attention_score = attention_score / torch.sum(attention_score, dim=1).unsqueeze(1).expand_as(attention_score)

#         # value: B, L, C
#         value = self.activate(self.fc_value(listener_feature))
#         context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(dim=1)

#         # context: B, C
#         return attention_score, context


class Speller(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, context_dim=256, num_layers=2, embedding_size=256, lockdropout=0.0, tie_weight=True, pre_train=False, use_gumbel=True):
        super(Speller, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.pre_train = pre_train 
        self.use_gumbel = use_gumbel 

        self.attention = AttentionQK(512, hidden_dim)
        # self.fc1 = nn.Linear(embedding_size + context_dim, hidden_dim)
        # self.concat_transformer = nn.Sequential(self.fc1, nn.Tanh())
        self.rnn_unit = nn.LSTMCell # has to be a cell unit rather than a full layer
        self.embedding = nn.Embedding(num_classes, embedding_size)
        # self.fc = nn.Linear(embedding_size+context_dim, hidden_dim)
        # self.concat_transformer = nn.Sequential(self.fc, nn.LeakyReLU(negative_slope=0.2))
        self.classifier = nn.Linear(embedding_size, num_classes)
        self.character_distribution = nn.Sequential(nn.Linear(hidden_dim + context_dim, embedding_size),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         self.classifier)

        if tie_weight:
            self.classifier.weight = self.embedding.weight

        self.rnn_layers = nn.ModuleList()
        self.rnn_init_h = nn.ParameterList()
        self.rnn_init_c = nn.ParameterList()
        self.rnn_layers.append(self.rnn_unit(embedding_size + context_dim, hidden_dim))
        # self.rnn_init_h.append(nn.Parameter(torch.zeros(1, hidden_dim)))
        # self.rnn_init_c.append(nn.Parameter(torch.zeros(1, hidden_dim)))

        # self.rnn_layers.append(self.rnn_unit(hidden_dim, hidden_dim))
        # self.rnn_init_h.append(nn.Parameter(torch.rand(1, hidden_dim)))
        # self.rnn_init_c.append(nn.Parameter(torch.rand(1, hidden_dim)))

        # # self.rnn_layers.append(self.rnn_unit(hidden_dim, hidden_dim))
        # # self.rnn_init_h.append(nn.Parameter(torch.rand(1, hidden_dim)))
        # # self.rnn_init_c.append(nn.Parameter(torch.rand(1, hidden_dim)))
        # self.rnn_layers.append(self.rnn_unit(hidden_dim, embedding_size))
        # self.rnn_init_h.append(nn.Parameter(torch.rand(1, embedding_size)))
        # self.rnn_init_c.append(nn.Parameter(torch.rand(1, embedding_size)))
        for i in range(num_layers):
        #     # if i == num_layers -1:
        #     #     self.rnn_layers.append(self.rnn_unit(hidden_dim, embedding_size))
        #     #     self.rnn_init_h.append(nn.Parameter(torch.rand(1, embedding_size)))
        #     #     self.rnn_init_c.append(nn.Parameter(torch.rand(1, embedding_size)))
            if i != 0:
                self.rnn_layers.append(self.rnn_unit(hidden_dim, hidden_dim))
            self.rnn_init_h.append(nn.Parameter(torch.rand(1, hidden_dim)))
            self.rnn_init_c.append(nn.Parameter(torch.rand(1, hidden_dim)))

    def get_init_state(self, batch_size):
        '''
        params:
            batch_size: int
        outputs:
            hidden:         list of initial hidden state: [batch_size, hidden_dim]
            cell:           list of initial cell value:   [batch_size, hidden_dim]
            output_word:    LongTensor of shape [batch_size, ]

        '''
        hidden = [h.repeat(batch_size, 1) for h in self.rnn_init_h]
        cell = [c.repeat(batch_size, 1) for c in self.rnn_init_c]
        output_word = Variable(hidden[-1].data.new(batch_size).long().fill_(0))
        # print(output_word)
        return (hidden, cell), output_word
        
    def forward(self, hs, state_len, max_iters, golden=None, teacher_force=0.9, dropout=[]):
        if golden is None:
            teacher_force = 0.0
        teacher = True if np.random.random_sample() < teacher_force and self.training else False
        batch_size = hs.size(0)
        init_state, output_word = self.get_init_state(batch_size)

        dropout_masks = []
        if dropout and self.training:
            # extract the hidden state from first layer
            hts = init_state[0]
            for i in range(self.num_layers):
                mask = hts[i].data.new(hts[i].size(0), hts[i].size(1)).bernoulli_(1-dropout[i]) / (1 - dropout[i])
                dropout_masks.append(Variable(mask, requires_grad=False))

        raw_pred_seq = []
        attention_score = []
        for step in range(golden.size(1) if golden is not None and self.training else max_iters):
            raw_pred, init_state, alpha = self.predict_one(hs, state_len, output_word, init_state, dropout_masks=dropout_masks)
            attention_score.append(alpha)
            raw_pred_seq.append(raw_pred)

            teacher = True if np.random.random_sample() < teacher_force and self.training else False
            if teacher:
                output_word = golden[:, step]
            else:
                output_word = torch.max(raw_pred, dim=-1)[1]

        return torch.stack(raw_pred_seq, dim=1), torch.stack(attention_score, dim=1)     

    def predict_one(self, hs, state_len, yt, ht, dropout_masks=None):
        previous_word_emb = self.embedding(yt)
        # print(previous_word_emb)
        # previous_word_emb = embedded_dropout(self.embedding, yt, dropout=0.1 if self.training else 0)
        # [batch_size, ] ===> [batch_size, embedding_size]
        hidden, cell = ht[0], ht[1]
        last_layer_ht = hidden[-1]
        # [batch_size, 1, hidden_dim]
        alpha, context = self.attention(hs, last_layer_ht, state_len)

        if self.pre_train:
            context = Variable(torch.zeros(context.size()), requires_grad=False).cuda()

        input_x = torch.cat((previous_word_emb, context), dim=1)
        # input_x = self.concat_transformer(input_x)

        new_hidden, new_cell = [None] * len(self.rnn_layers), [None] * len(self.rnn_layers)
        for l, rnn in enumerate(self.rnn_layers):
            new_hidden[l], new_cell[l] = rnn(input_x, (hidden[l], cell[l]))
            if dropout_masks and self.training:
                input_x = new_hidden[l] * dropout_masks[l]
            else: 
                input_x = new_hidden[l]    
        rnn_output = new_hidden[-1]

        concat_feature = torch.cat([rnn_output, context], dim=1)
        yt = self.character_distribution(concat_feature)

        if self.use_gumbel and self.training:
            yt = self.gumbel_noise(yt)
        return yt, (new_hidden, new_cell), alpha


    def gumbel_noise(self, yt):
        shape = yt.shape
        gbl = torch.distributions.gumbel.Gumbel(0, 0.1)
        noise = Variable(gbl.sample(shape), requires_grad=False).cuda()
        yt = yt + noise
        del noise, gbl
        return yt 


# class Speller2(nn.Module):
#     def __init__(self, n_classes, speller_hidden_dim, speller_rnn_layer, attention, context_dim):
#         super(Speller2, self).__init__()
#         self.n_classes = n_classes
#         self.rnn_unit = nn.LSTMCell

#         # self.rnn_layer = self.rnn_unit(output_class_dim + speller_hidden_dim, speller_hidden_dim,
#         #                                num_layers=speller_rnn_layer)

#         self.rnn_layer = torch.nn.ModuleList()
#         self.rnn_inith = torch.nn.ParameterList()
#         self.rnn_initc = torch.nn.ParameterList()
#         self.rnn_layer.append(self.rnn_unit(speller_hidden_dim + context_dim, speller_hidden_dim))
#         for i in range(speller_rnn_layer):
#             if i != 0:
#                 self.rnn_layer.append(self.rnn_unit(speller_hidden_dim, speller_hidden_dim))
#             self.rnn_inith.append(torch.nn.Parameter(torch.rand(1, speller_hidden_dim)))
#             self.rnn_initc.append(torch.nn.Parameter(torch.rand(1, speller_hidden_dim)))

#         self.attention = attention

#         # char embedding
#         self.embed = nn.Embedding(n_classes, speller_hidden_dim)

#         # prob output layers
#         self.fc = nn.Linear(speller_hidden_dim + context_dim, speller_hidden_dim)
#         self.activate = torch.nn.LeakyReLU(negative_slope=0.2)
#         self.unembed = nn.Linear(speller_hidden_dim, n_classes)
#         self.unembed.weight = self.embed.weight
#         self.character_distribution = nn.Sequential(self.fc, self.activate, self.unembed)

#     def forward(self, listener_feature, seq_sizes, max_iters, ground_truth=None, teacher_force_rate=0.9, dropout=[]):
#         if ground_truth is None:
#             teacher_force_rate = 0
#         teacher_force = True if np.random.random_sample() < teacher_force_rate and self.training else False
#         batch_size = listener_feature.size()[0]

#         state, output_word = self.get_initial_state(batch_size)

#         # dropouts
#         dropout_masks = []
#         if dropout and self.training:
#             h = state[0][0] # B, C
#             n_layers = len(state[0])
#             for i in range(n_layers):
#                 mask = h.data.new(h.size(0), h.size(1)).bernoulli_(1 - dropout[i]) / (1 - dropout[i])
#                 dropout_masks.append(Variable(mask, requires_grad=False))

#         raw_pred_seq = []
#         attention_record = []
#         for step in range(ground_truth.size(1) if ground_truth is not None else max_iters):

#             # print("last_output_word_forward", idx2chr[output_word.data[0]])
#             attention_score, raw_pred, state = self.run_one_step(listener_feature, seq_sizes, output_word, state, dropout_masks=dropout_masks)

#             attention_record.append(attention_score)
#             raw_pred_seq.append(raw_pred)

#             # Teacher force - use ground truth as next step's input
#             if teacher_force:
#                 output_word = ground_truth[:, step]
#             else:
#                 output_word = torch.max(raw_pred, dim=1)[1]

#         return torch.stack(raw_pred_seq, dim=1), attention_record

#     def run_one_step(self, listener_feature, seq_sizes, last_output_word, state, dropout_masks=None):
#         output_word_emb = self.embed(last_output_word)

#         # get attention context
#         hidden, cell = state[0], state[1]
#         last_rnn_output = hidden[-1]  # last layer
#         # print("last_rnn_output", last_rnn_output)
#         # print("listener_feature", listener_feature)
#         attention_score, context = self.attention(last_rnn_output, listener_feature, seq_sizes)

#         # run speller rnns for one time step
#         rnn_input = torch.cat([output_word_emb, context], dim=1)
#         new_hidden, new_cell = [None] * len(self.rnn_layer), [None] * len(self.rnn_layer)
#         for l, rnn in enumerate(self.rnn_layer):
#             new_hidden[l], new_cell[l] = rnn(rnn_input, (hidden[l], cell[l]))
#             if dropout_masks:
#                 rnn_input = new_hidden[l] * dropout_masks[l]
#             else:
#                 rnn_input = new_hidden[l]
#         rnn_output = new_hidden[-1]  # last layer

#         # make prediction
#         concat_feature = torch.cat([rnn_output, context], dim=1)
#         #             print("concat_feature.size()", concat_feature.size())
#         raw_pred = self.character_distribution(concat_feature)
#         #             print("raw_pred.size()", raw_pred.size())
#         return attention_score, raw_pred, [new_hidden, new_cell]

#     def get_initial_state(self, batch_size=32):
#         hidden = [h.repeat(batch_size, 1) for h in self.rnn_inith]
#         cell = [c.repeat(batch_size, 1) for c in self.rnn_initc]
#         # <sos> (same vocab as <eos>)
#         output_word = Variable(hidden[0].data.new(batch_size).long().fill_(0))
#         # print(output_word)
#         return [hidden, cell], output_word


class LAS(nn.Module):
    def __init__(self, num_classes, pre_train=False):
        super(LAS, self).__init__()
        self.listener = Listener()
        self.speller = Speller(num_classes, pre_train=pre_train)
        # self.attention = AttentionQK(key_query_dim=128, speller_query_dim=256, listener_feature_dim=512, context_dim=128)
        # self.speller = Speller(33, 256, 3, self.attention, context_dim=128)

        self._hs = None
        self._state_len = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_x, seq_len, labels, teacher_force=0.9, max_iters=250):
        hs, state_len = self.listener(input_x, seq_len)
        outputs, attention_score = self.speller(hs, state_len, max_iters, labels, teacher_force=teacher_force, dropout=[0.2, 0.2]) 
        return outputs, attention_score

    def get_init_state(self, input_x, seq_len):
        self._hs, self._state_len = self.listener(input_x, seq_len)
        init_state, output_word = self.speller.get_init_state(batch_size=1)
        return init_state, output_word.data[0]

    def generate(self, input_x, yts, last_states):
        new_states, raw_preds, attention_score = [], [], []
        
        for yt, last_state in zip(yts, last_states):
            yt = Variable(self._hs.data.new(1).long().fill_(int(yt)))
            raw_pred, new_state, alpha = self.speller.predict_one(self._hs, self._state_len, yt, last_state)
            new_states.append(new_state)
            raw_preds.append(self.softmax(raw_pred).squeeze().data.cpu().numpy())
            attention_score.append(alpha)

        return new_states, raw_preds, attention_score

