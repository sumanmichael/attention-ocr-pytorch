import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionDecoderV2(nn.Module):
    """
        Use seq to seq model to modify the calculation method of attention weight
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttentionDecoderV2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # test
        self.vat = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # Word embedding on the previous output
        embedded = self.dropout(embedded)
        # test
        batch_size = encoder_outputs.shape[1]
        alpha = hidden + encoder_outputs  # Feature fusion +/concat can actually be used
        alpha = alpha.view(-1, alpha.shape[-1])
        attn_weights = self.vat(
            torch.tanh(alpha))  # Reduce encoder_output: batch*seq*features to reduce the dimension of features to 1
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim=2)

        # Find the weight of the last output and hidden state
        # attn_weights = F.softmax(
        #     self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute(
                                        (1, 0, 2)))  # Matrix multiplication，bmm（8×1×56，8×56×256）=8×1×256
        if len(embedded.shape)==1:
            embedded = embedded.unsqueeze(0)
        output = torch.cat((embedded, attn_applied.squeeze(1)),
                           1)  # The last output and attention feature, make a linear + GRU
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)  # Finally output a probability
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


class DecoderV2(pl.LightningModule):
    '''
        decoder from image features
    '''

    def __init__(self, nh=256, nclass=13, dropout_p=0.1):
        super(DecoderV2, self).__init__()
        self.hidden_size = nh
        self.decoder = AttentionDecoderV2(nh, nclass, dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result
