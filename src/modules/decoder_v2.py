import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionDecoder(pl.LightningModule):
    """
        Use seq to seq model to modify the calculation method of attention weight
    """

    def __init__(self, encoder_hidden_size, decoder_hidden_size, output_size, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.lstm = nn.LSTM(self.output_size * 2, self.decoder_hidden_size, num_layers=2)
        self.W_o = nn.Linear(self.output_size, self.output_size)
        self.W_c = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.output_size)
        self.W_h = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
        self.W_v = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.beta_T = nn.Linear(self.decoder_hidden_size, 1)

        self.rnn_state = None
        self.enc_outs = None

    def forward(self, y_prev, o_prev, hidden_prev):
        """
            e_it = β^T tanh(W_h x h_(i−1) + W_v x ˜v_t)
            α_t = softmax(e_t)
            c_i = sum(α_it x v_t)
            h_t = RNN(h_(t−1), [y_(t−1);o_(t−1)])
            o_t = tanh(W_c[h_t;c_t])
            p(y_(t+1) | y_1, ..., y_t, V˜ ) = softmax(W_o x o_t)
        """

        whh = self.W_h(hidden_prev)
        wvv = self.W_v(self.enc_outs)
        th = torch.tanh(whh + wvv)
        e = self.beta_T(th)
        alpha = torch.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.bmm(alpha, self.enc_outs)
        rnn_inp = torch.cat((y_prev, o_prev), dim=2)

        hidden, self.rnn_state = self.lstm(rnn_inp, self.rnn_state)

        hc = torch.cat((hidden, context), dim=2)
        o = torch.tanh(self.W_c(hc))
        y = F.softmax(self.W_o(o), dim=0)

        return y, o, hidden

    def init_hidden(self, batch_size):
        result = Variable(torch.randn(batch_size, 1, self.decoder_hidden_size))
        return result

    def init_rnn_state(self):
        return torch.randn(2, 1, self.decoder_hidden_size).to(self.device), \
               torch.randn(2, 1, self.decoder_hidden_size).to(self.device)
