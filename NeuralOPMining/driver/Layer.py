import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math


def default_init(tensor):
    if tensor.ndimension() == 1:
        nn.init.constant_(tensor, val=0.0)
    else:
        nn.init.xavier_normal_(tensor)

    return tensor

def reset_bias_with_orthogonal(bias):
    bias_temp = torch.nn.Parameter(torch.FloatTensor(bias.size()[0], 1))
    nn.init.orthogonal(bias_temp)
    bias_temp = bias_temp.view(-1)
    bias.data = bias_temp.data

def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    scale = 1.0 / (1.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings

def drop_biinput_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)

def get_tensor_np(t):
    return t.data.cpu().numpy()


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def block_orth_normal_initializer(input_size, output_size):
    weight = []
    for o in output_size:
        for i in input_size:
            param = torch.FloatTensor(o, i)
            torch.nn.init.orthogonal(param)
            weight.append(param)
    return torch.cat(weight)


class DropoutLayer(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0):
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.drop_mask = torch.FloatTensor(self.input_size).fill_(1 - self.dropout_rate)
        self.drop_mask = Variable(torch.bernoulli(self.drop_mask), requires_grad=False)
        if torch.cuda.is_available():
            self.drop_mask = self.drop_mask.cuda()

    def reset_dropout_mask(self, batch_size):
        self.drop_mask = torch.FloatTensor(batch_size, self.input_size).fill_(1 - self.dropout_rate)
        self.drop_mask = Variable(torch.bernoulli(self.drop_mask), requires_grad=False)
        if torch.cuda.is_available():
            self.drop_mask = self.drop_mask.cuda()

    def forward(self, x):
        if self.training:
            return torch.mul(x, self.drop_mask)
        else:  # eval
            return x * (1.0 - self.dropout_rate)


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal(self.linear.weight)
        if self.bias:
            reset_bias_with_orthogonal(self.linear.bias)


    def forward(self, x):
        return self.linear(x)

class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        nn.init.orthogonal(self.linear.weight)
        reset_bias_with_orthogonal(self.linear.bias)


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):  #
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)  # 1e-3 is ok, because variance and std.
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1, size[0], size[1])

class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'

class LSTM(nn.LSTM):
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                for i in range(4):
                    nn.init.orthogonal(self.__getattr__(name)[self.hidden_size*i:self.hidden_size*(i+1),:])
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)

class MyLSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        self._all_weights = []
        for layer in range(num_layers):
            layer_params = (self.fcells[layer].weight_ih, self.fcells[layer].weight_hh, \
                            self.fcells[layer].bias_ih, self.fcells[layer].bias_hh)
            suffix = ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            if self.bidirectional:
                layer_params = (self.bcells[layer].weight_ih, self.bcells[layer].weight_hh, \
                                self.bcells[layer].bias_ih, self.bcells[layer].bias_hh)
                suffix = '_reverse'
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            if self.bidirectional:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '_reverse')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '_reverse')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + 2 * self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))
            else:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)


    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = MyLSTM._forward_rnn(cell=self.fcells[layer], \
                input=input, masks=masks, initial=initial, drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = MyLSTM._forward_brnn(cell=self.bcells[layer], \
                    input=input, masks=masks, initial=initial, drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)

class MyHighwayLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyHighwayLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=6 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=5 * hidden_size,
                                   bias=False)
        self.reset_parameters()  # reset all the param in the MyLSTMCell

    def reset_parameters(self):
        weight_ih = block_orth_normal_initializer([self.input_size, ], [self.hidden_size] * 6)
        self.linear_ih.weight.data.copy_(weight_ih)

        weight_hh = block_orth_normal_initializer([self.hidden_size, ], [self.hidden_size] * 5)
        self.linear_hh.weight.data.copy_(weight_hh)
        # nn.init.constant(self.linear_hh.weight, 1.0)
        # nn.init.constant(self.linear_ih.weight, 1.0)

        nn.init.constant(self.linear_ih.bias, 0.0)

    def forward(self, x, mask=None, hx=None, dropout=None):
        assert mask is not None and hx is not None
        _h, _c = hx
        _x = self.linear_ih(x)  # compute the x
        preact = self.linear_hh(_h) + _x[:, :self.hidden_size * 5]

        i, f, o, t, j = preact.chunk(chunks=5, dim=1)
        i, f, o, t, j = F.sigmoid(i), F.sigmoid(f + 1.0), F.sigmoid(o), F.sigmoid(t), F.tanh(j)
        k = _x[:, self.hidden_size * 5:]

        c = f * _c + i * j
        c = mask * c + (1.0 - mask) * _c

        h = t * o * F.tanh(c) + (1.0 - t) * k
        if dropout is not None:
            h = dropout(h)
        h = mask * h + (1.0 - mask) * _h
        return h, c


class HBiLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, cuda_id = ""):
        super(HBiLSTM, self).__init__()
        self.batch_size = 1
        self.cuda_id = cuda_id
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1  # this is a BiLSTM with Highway
        self.bilstm = nn.LSTM(self.in_dim, self.hidden_dim, num_layers=self.num_layers, \
                              batch_first=True, bidirectional=True)
        self.in_dropout_layer = DropoutLayer(in_dim, 0.1)
        self.out_dropout_layer = DropoutLayer(2 * hidden_dim, 0.1)
        # Highway gate layer T in the Highway formula
        self.gate_layer = nn.Linear(self.in_dim, self.hidden_dim * 2)
        # self.dropout_layer = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        print("Initing W .......")
        nn.init.orthogonal(self.bilstm.all_weights[0][0])
        nn.init.orthogonal(self.bilstm.all_weights[0][1])
        nn.init.orthogonal(self.bilstm.all_weights[1][0])
        nn.init.orthogonal(self.bilstm.all_weights[1][1])
        if self.bilstm.bias is True:
            print("Initing bias......")
            a = np.sqrt(2 / (1 + 600)) * np.sqrt(3)
            nn.init.uniform(self.bilstm.all_weights[0][2], -a, a)
            nn.init.uniform(self.bilstm.all_weights[0][3], -a, a)
            nn.init.uniform(self.bilstm.all_weights[1][2], -a, a)
            nn.init.uniform(self.bilstm.all_weights[1][3], -a, a)

    def __init_hidden(self):
        if self.cuda_id:
            return (Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim)))

    def forward(self, x, batch_size, x_lengths):
        self.batch_size = batch_size
        hidden = self.__init_hidden()

        source_x = x
        if self.training:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = self.in_dropout_layer(x)  # input dropout
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu().numpy() \
                if self.cuda_id else x_lengths.numpy(), batch_first=True)
        x, hidden = self.bilstm(x, hidden)  # [Batch, T, H], batch first = True
        if self.training:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = self.out_dropout_layer(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu().numpy() \
                if self.cuda_id else x_lengths.numpy(), batch_first=True)
        lstm_out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        source_x, _ = torch.nn.utils.rnn.pad_packed_sequence(source_x, batch_first=True)

        batched_output = []
        for i in range(batch_size):
            ith_lstm_output = lstm_out[i][:output_lengths[i]]  # [actual size, hidden_dim]
            ith_source_x = source_x[i][:output_lengths[i]]

            # r gate: r = sigmoid(x*W + b)
            information_source = self.gate_layer(ith_source_x)
            transformation_layer = F.sigmoid(information_source)
            # formula Y = H * T + x * C
            allow_transformation = torch.mul(transformation_layer, ith_lstm_output)

            # carry gate layer in the formula
            carry_layer = 1 - transformation_layer
            # the information_source compare to the source_x is for the same size of x,y,H,T
            allow_carry = torch.mul(information_source, carry_layer)
            # allow_carry = torch.mul(source_x, carry_layer)
            information_flow = torch.add(allow_transformation, allow_carry)

            padding = nn.ConstantPad2d((0, 0, 0, output_lengths[0] - information_flow.size()[0]), 0.0)
            information_flow = padding(information_flow)
            batched_output.append(information_flow)

        information_flow = torch.stack(batched_output)
        if self.training:
            information_flow = drop_sequence_sharedmask(information_flow, 0.1)
        information_flow = torch.nn.utils.rnn.pack_padded_sequence(information_flow, x_lengths.cpu().numpy() \
            if self.cuda_id else x_lengths.numpy(), batch_first=True)

        return information_flow, hidden


class HighwayBiLSTM(nn.Module):
    """A module that runs multiple steps of HighwayBiLSTM."""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(HighwayBiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells, self.f_dropout, self.f_hidden_dropout = [], [], []
        self.bcells, self.b_dropout, self.b_hidden_dropout = [], [], []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.fcells.append(MyHighwayLSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            self.f_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.f_hidden_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            if self.bidirectional:
                self.bcells.append(MyHighwayLSTMCell(input_size=hidden_size, hidden_size=hidden_size))
                self.b_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
                self.b_hidden_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
        self.fcells, self.bcells = nn.ModuleList(self.fcells), nn.ModuleList(self.bcells)
        self.f_dropout, self.b_dropout = nn.ModuleList(self.f_dropout), nn.ModuleList(self.b_dropout)

    def reset_dropout_layer(self, batch_size):
        for layer in range(self.num_layers):
            self.f_dropout[layer].reset_dropout_mask(batch_size)
            if self.bidirectional:
                self.b_dropout[layer].reset_dropout_mask(batch_size)

    @staticmethod
    def _forward_rnn(cell, gate, input, masks, initial, drop_masks=None, hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input[time], mask=masks[time], hx=hx, dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, gate, input, masks, initial, drop_masks=None, hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input[time], mask=masks[time], hx=hx, dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)  # transpose: return the transpose matrix
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()

        self.reset_dropout_layer(batch_size)  # reset the dropout each batch forward

        masks = masks.expand(-1, -1, self.hidden_size)  # expand: -1 means not expand that dimension
        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)  # h0, c0

        h_n, c_n = [], []
        for layer in range(self.num_layers):
            # hidden_mask, hidden_drop = None, None
            hidden_mask, hidden_drop = self.f_dropout[layer], self.f_hidden_dropout[layer]
            layer_output, (layer_h_n, layer_c_n) = HighwayBiLSTM._forward_rnn(cell=self.fcells[layer], \
                            gate=None, input=input, masks=masks, initial=initial, \
                            drop_masks=hidden_mask, hidden_drop=hidden_drop)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
            if self.bidirectional:
                hidden_mask, hidden_drop = self.b_dropout[layer], self.b_hidden_dropout[layer]
                blayer_output, (blayer_h_n, blayer_c_n) = HighwayBiLSTM._forward_brnn(cell=self.bcells[layer], \
                            gate=None, input=layer_output, masks=masks, initial=initial, \
                            drop_masks=hidden_mask, hidden_drop=hidden_drop)
                h_n.append(blayer_h_n)
                c_n.append(blayer_c_n)

            input = blayer_output if self.bidirectional else layer_output

        h_n, c_n = torch.stack(h_n, 0), torch.stack(c_n, 0)
        if self.batch_first:
            input = input.transpose(1, 0)  # transpose: return the transpose matrix
        return input, (h_n, c_n)