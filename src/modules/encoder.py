from functools import reduce
from operator import __add__

import pytorch_lightning as pl
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self._zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
                                                [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in
                                                 self.kernel_size[::-1]]))

    def forward(self, input):
        input = self._zero_pad_2d(input)
        return self._conv_forward(input, self.weight, self.bias)


class CNN(pl.LightningModule):
    '''
        CNN+BiLstm does feature extraction
    '''

    def __init__(self, imgH, nc, nh):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        #     TODO: Bias in Conv2D?
        self.cnn = nn.Sequential(  # 1x32x512
            Conv2dSamePadding(nc, 64, 3, 1, 0), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 64x32x256
            Conv2dSamePadding(64, 128, 3, 1, 0), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x8x128
            Conv2dSamePadding(128, 256, 3, 1, 0), nn.BatchNorm2d(256), nn.ReLU(True),  # 256x8x128
            Conv2dSamePadding(256, 256, 3, 1, 0), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x128
            Conv2dSamePadding(256, 512, 3, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),  # 512x4x128
            Conv2dSamePadding(512, 512, 3, 1, 0), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x128
            Conv2dSamePadding(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)))  # 512x1x128
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))

    def forward(self, input):
        # conv features
        # conv = self.cnn(input)

        x = input
        for layer in self.cnn:
            # print(x.size(), layer)
            x = layer(x)
        conv = x
        # print(conv.size())

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features calculate
        encoder_outputs = self.rnn(conv)  # seq * batch * n_classes// 25 × batchsize × 256（Number of hidden nodes）

        return encoder_outputs