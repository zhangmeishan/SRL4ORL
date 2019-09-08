import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from driver.Layer import *


def add_pos_embedding(x, min_timescale=1.0, max_timescale=1.0e4):
    batch, length, channels = list(x.size())
    assert (channels % 2 == 0)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1.))
    position = torch.arange(0, length).float()
    inv_timescales = torch.arange(0, num_timescales).float()
    if x.is_cuda:
        position = position.cuda()
        inv_timescales = inv_timescales.cuda()

    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    scaled_time = position.unsqueeze(1).expand(
        length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
    # scaled time is now length x num_timescales
    # length x channels
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)
    signal = signal.unsqueeze(0).expand(batch, length, channels)

    return torch.autograd.Variable(signal)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        """
        :type attn_mask: torch.FloatTensor
        :param attn_mask: Mask of the attention.
            3D tensor with shape [batch_size, time_step_key, time_step_value]
        """
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill(attn_mask, -1e18)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = LayerNormalization(size)
        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        #self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal(self.w_1.weight)
        reset_bias_with_orthogonal(self.w_1.bias)

        nn.init.orthogonal(self.w_2.weight)
        reset_bias_with_orthogonal(self.w_2.bias)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MyMultiHeadAttention(nn.Module):
    ''' MyMulti-Head Attention module '''

    def __init__(self, head_count, model_dim):
        super(MyMultiHeadAttention, self).__init__()
        self.n_head = head_count
        d_k = d_v = model_dim // head_count

        self.d_k = d_k
        self.d_v = d_v

        self.qkv_combined = nn.Linear(model_dim, 3 * model_dim)
        self.attention = ScaledDotProductAttention(model_dim)
        self.proj = nn.Linear(head_count*d_v, model_dim, bias=True)  # kiro

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal(self.qkv_combined.weight)
        reset_bias_with_orthogonal(self.qkv_combined.bias)
        nn.init.orthogonal(self.proj.weight)
        reset_bias_with_orthogonal(self.proj.bias)

    def forward(self, q, k, v, mask=None, attn_bias=None):
        residual = q

        combined_qkv = self.qkv_combined(q)
        q, k, v = torch.chunk(combined_qkv, 3, dim=-1)
        q_s, k_s, v_s = torch.chunk(q, self.n_head, dim=-1), \
                        torch.chunk(k, self.n_head, dim=-1), \
                        torch.chunk(v, self.n_head, dim=-1)
        q_s, k_s, v_s = torch.stack(q_s, dim=0), torch.stack(k_s, dim=0), torch.stack(v_s, dim=0)
        q_s, k_s, v_s = q_s.view(-1, q_s.size()[-2], q_s.size()[-1]), \
                        k_s.view(-1, k_s.size()[-2], k_s.size()[-1]), \
                        v_s.view(-1, v_s.size()[-2], v_s.size()[-1])

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=mask.repeat(self.n_head, 1, 1))
        outputs = outputs.view(self.n_head, -1, outputs.size()[-2], outputs.size()[-1])
        outputs = torch.chunk(outputs, self.n_head, dim=0)
        outputs = torch.cat(outputs, dim=-1).squeeze(dim=0)
        # project back to residual size
        outputs = self.proj(outputs)

        return outputs, attns

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, head_count, model_dim):
        super(MultiHeadAttention, self).__init__()

        self.n_head = head_count
        d_k = d_v = model_dim // head_count

        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(head_count, model_dim, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(head_count, model_dim, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(head_count, model_dim, d_v))

        self.attention = ScaledDotProductAttention(model_dim)
        self.proj = nn.Linear(head_count*d_v, model_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal(self.w_qs)
        nn.init.orthogonal(self.w_ks)
        nn.init.orthogonal(self.w_vs)

        nn.init.orthogonal(self.proj.weight)
        reset_bias_with_orthogonal(self.proj.bias)


    def forward(self, q, k, v, mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head


        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=mask.repeat(n_head, 1, 1))

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)

        return outputs, attns


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """
    def __init__(self, head_count, model_dim):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, \
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, \
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, \
                                      head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)

        self.final_linear = nn.Linear(model_dim, model_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal(self.linear_keys.weight)
        reset_bias_with_orthogonal(self.linear_keys.bias)

        nn.init.orthogonal(self.linear_values.weight)
        reset_bias_with_orthogonal(self.linear_values.bias)

        nn.init.orthogonal(self.linear_query.weight)
        reset_bias_with_orthogonal(self.linear_query.bias)

        nn.init.orthogonal(self.final_linear.weight)
        reset_bias_with_orthogonal(self.final_linear.bias)

    def _split_heads(self, x):

        batch_size = x.size(0)

        return x.view(batch_size, -1, self.head_count, self.dim_per_head) \
            .transpose(1, 2)

    def _combine_heads(self, x):

        """:param x: [batch_size * head_count, seq_len, dim_per_head]"""
        seq_len = x.size(2)

        return x.transpose(1, 2).contiguous() \
            .view(-1, seq_len, self.head_count * self.dim_per_head)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.

        key_up = self._split_heads(self.linear_keys(key)) # [batch_size, num_head, seq_len, dim_head]
        value_up = self._split_heads(self.linear_values(value))


        query_up = self._split_heads(self.linear_query(query))

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        context = self._combine_heads(torch.matmul(attn, value_up))

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn.view(batch_size, head_count, \
                  query_len, key_len)[:, 0, :, :].contiguous()
        # END CHECK
        return output, top_attn


class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super(EncoderBlock, self).__init__()

        #self.slf_attn = MultiHeadAttention(n_head, d_model)
        #self.slf_attn = MyMultiHeadAttention(n_head, d_model)
        self.slf_attn = MultiHeadedAttention(n_head, d_model)

        self.layer_norm = LayerNormalization(d_model)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)


    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.pos_ffn(enc_input)
        ##The below norlamization is very important!!
        input_norm = self.layer_norm(input_norm)
        enc_output, enc_slf_attn = self.slf_attn(input_norm, input_norm, input_norm, mask=slf_attn_mask)
        out = self.dropout(enc_output) + enc_input

        return out
