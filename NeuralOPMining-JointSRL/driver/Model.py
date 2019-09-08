from driver.Layer import *
from driver.Attention import *
from driver.CRF import *
from data.Vocab import *

class BiLSTMModel(nn.Module):
    """ Constructs the network and builds the following Theano functions:
        - pred_function: Takes input and mask, returns prediction.
        - loss_function: Takes input, mask and labels, returns the final cross entropy loss (scalar).
    """
    def __init__(self, vocab, config, pretrained_embedding, srlout_dim):
        super(BiLSTMModel, self).__init__()
        self.config = config
        self.PAD = vocab.PAD
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=self.PAD)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=self.PAD)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        # {0/pad, 1/predict, 2/other}
        self.predicate_embed = nn.Embedding(3, config.predict_dims, padding_idx=self.PAD)
        nn.init.normal(self.predicate_embed.weight, 0.0, 1.0 / (config.predict_dims ** 0.5))

        self.srlproj = nn.Linear(srlout_dim, config.hidden_dims)
        nn.init.normal(self.srlproj.weight, 0.0, 1.0 / (srlout_dim ** 0.5))
        nn.init.normal(self.srlproj.bias, 0.0, 1.0 / (srlout_dim ** 0.5))
        self.active = nn.Tanh()

        self.input_dims = config.word_dims + config.hidden_dims + config.predict_dims

        # Initialize BiLSTM
        self.bilstm = MyLSTM(
            input_size=self.input_dims,
            hidden_size=config.lstm_hiddens,  # // 2 for MyLSTM
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden
        )
        self.outlayer = nn.Linear(2 * config.lstm_hiddens, vocab.label_size, bias=False)
        nn.init.normal(self.outlayer.weight, 0.0, 1.0 / ((2 * config.lstm_hiddens) ** 0.5))


    def forward(self, words, extwords, predicts, inmasks, srl_outputs):
        # self.highway_lstm1 = HBiLSTM(self.word_embedding_dim * 2, self.lstm_hidden_size // 2, batch_size, self.cuda_id)
        # self.hidden = self.__init_hidden(batch_size)  # clear the hidden state of the LSTM
        # sentence length, mini batch size, input size
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_predict_embed = self.predicate_embed(predicts)
        x_srl_inputs = self.active(self.srlproj(srl_outputs))

        if self.training:
            x_embed, x_predict_embed = drop_biinput_independent(x_embed, x_predict_embed, self.config.dropout_emb)

        embeddings = torch.cat((x_embed, x_srl_inputs, x_predict_embed), 2)

        lstm_out, _ = self.bilstm(embeddings, inmasks)
        lstm_out = lstm_out.transpose(1, 0)

        label_scores = self.outlayer(lstm_out)
        #self.label_scores = F.log_softmax(self.label_scores, dim=2)
        return label_scores

    def compute_loss(self, output, answer, outmasks):
        # output: [B, T, L], answer: [B, T], mask: [B, T, L]
        # print answer
        B, T, L = output.size()
        output = output.view(B * T, L)
        target = answer.view(B * T)
        loss = F.cross_entropy(output, target, ignore_index=self.PAD)
        return loss

    def decode(self, label_scores, inmask):
        label_scores = label_scores.data.cpu().numpy()
        predict_labels = np.argmax(label_scores, axis=2)

        return predict_labels

    def save(self, filepath):
        """ Save model parameters to file.
        """
        torch.save(self.state_dict(), filepath)
        print('Saved model to: {}'.format(filepath))

    def load(self, filepath):
        """ Load model parameters from file.
        """
        self.load_state_dict(torch.load(filepath))
        print('Loaded model from: {}'.format(filepath))


class BiLSTMCRFModel(nn.Module):
    """ Constructs the network and builds the following Theano functions:
        - pred_function: Takes input and mask, returns prediction.
        - loss_function: Takes input, mask and labels, returns the final cross entropy loss (scalar).
    """
    def __init__(self, vocab, config, pretrained_embedding, srlout_dim):
        super(BiLSTMCRFModel, self).__init__()
        self.config = config
        self.PAD = vocab.PAD
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=self.PAD)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=self.PAD)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        # {0/pad, 1/predict, 2/other}
        self.predicate_embed = nn.Embedding(3, config.predict_dims, padding_idx=self.PAD)
        nn.init.normal(self.predicate_embed.weight, 0.0, 1.0 / (config.predict_dims ** 0.5))

        self.srlproj = nn.Linear(srlout_dim, config.hidden_dims)
        nn.init.normal(self.srlproj.weight, 0.0, 1.0 / (srlout_dim ** 0.5))
        nn.init.normal(self.srlproj.bias, 0.0, 1.0 / (srlout_dim ** 0.5))
        self.active = nn.Tanh()

        self.input_dims = config.word_dims + config.hidden_dims + config.predict_dims

        # Initialize BiLSTM
        self.bilstm = MyLSTM(
            input_size=self.input_dims,
            hidden_size=config.lstm_hiddens,  # // 2 for MyLSTM
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden
        )
        #self.output_dims = vocab.label_size + 2
        self.outlayer = nn.Linear(2 * config.lstm_hiddens, vocab.label_size, bias=False)
        nn.init.normal(self.outlayer.weight, 0.0, 1.0 / ((2 * config.lstm_hiddens) ** 0.5))

        self.crf = CRF(vocab.label_size)


    def forward(self, words, extwords, predicts, inmasks, srl_outputs):
        # self.highway_lstm1 = HBiLSTM(self.word_embedding_dim * 2, self.lstm_hidden_size // 2, batch_size, self.cuda_id)
        # self.hidden = self.__init_hidden(batch_size)  # clear the hidden state of the LSTM
        # sentence length, mini batch size, input size
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_predict_embed = self.predicate_embed(predicts)
        x_srl_inputs = self.active(self.srlproj(srl_outputs))

        if self.training:
            x_embed, x_predict_embed = drop_biinput_independent(x_embed, x_predict_embed, self.config.dropout_emb)

        embeddings = torch.cat((x_embed, x_srl_inputs, x_predict_embed), 2)

        lstm_out, _ = self.bilstm(embeddings, inmasks)
        lstm_out = lstm_out.transpose(1, 0)

        self.label_scores = self.outlayer(lstm_out)

        #self.label_scores = F.log_softmax(self.label_scores, dim=2)
        return self.label_scores

    def compute_loss(self, output, answer, outmasks):
        # output: [B, T, L], answer: [B, T], mask: [B, T, L]
        # print answer
        output = output.transpose(1, 0).contiguous()
        answer = answer.transpose(1, 0).contiguous()
        outmasks = outmasks.transpose(1, 0).contiguous()
        total_loss = self.crf(output, answer, outmasks)

        num_words = outmasks.float().sum()
        total_loss = total_loss / num_words

        return total_loss

    def decode(self, label_scores, inmask):
        label_scores = label_scores.transpose(1, 0).contiguous()
        inmask = inmask.transpose(1, 0).contiguous()
        tag_seq = self.crf.decode(label_scores, inmask)

        return tag_seq

    def save(self, filepath):
        """ Save model parameters to file.
        """
        torch.save(self.state_dict(), filepath)
        print('Saved model to: {}'.format(filepath))

    def load(self, filepath):
        """ Load model parameters from file.
        """
        self.load_state_dict(torch.load(filepath))
        print('Loaded model from: {}'.format(filepath))

