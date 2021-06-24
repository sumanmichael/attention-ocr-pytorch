import torch
from torch import nn
from src.modules.encoder import Encoder
from src.modules.custom import Conv2dSamePadding, BatchNorm2d
import numpy as np
import pickle

def get_pytorch_lstm_weights_from_tensorflow(kernel, b, INPUT_SIZE, alpha=0.5):
    i, j, f, o = np.split(kernel[:INPUT_SIZE], 4, 1)
    kih = np.concatenate((i, f, j, o), axis=1).transpose((1, 0))
    i, j, f, o = np.split(kernel[INPUT_SIZE:], 4, 1)
    khh = np.concatenate((i, f, j, o), axis=1).transpose((1, 0))

    i, j, f, o = np.split(b, 4, 0)
    bih = np.concatenate((i, f, j, o), axis=0)

    w_ih = nn.Parameter(torch.tensor(kih))
    w_hh = nn.Parameter(torch.tensor(khh))
    b_ih = nn.Parameter(torch.tensor(bih) * alpha)
    b_hh = nn.Parameter(torch.tensor(bih) * (1-alpha))

    return w_ih, w_hh, b_ih, b_hh



d = pickle.load(open("wts.pkl", "rb"))
wt_map = {
    "0": 'conv_conv1/W',
    "3": 'conv_conv2/W',
    "6": 'conv_conv3/W',
    "7": ['conv_conv3/BatchNorm/gamma', 'conv_conv3/BatchNorm/beta',
          'conv_conv3/BatchNorm/moving_mean', 'conv_conv3/BatchNorm/moving_variance'],
    "9": 'conv_conv4/W',
    "12": 'conv_conv5/W',
    "13": ['conv_conv5/BatchNorm/gamma', 'conv_conv5/BatchNorm/beta',
           'conv_conv5/BatchNorm/moving_mean', 'conv_conv5/BatchNorm/moving_variance'],
    "15": 'conv_conv6/W',
    "18": 'conv_conv7/W',
    "19": ['conv_conv7/BatchNorm/gamma', 'conv_conv7/BatchNorm/beta',
           'conv_conv7/BatchNorm/moving_mean', 'conv_conv7/BatchNorm/moving_variance'],
    "enc_rnn": ["bidirectional_rnn/fw/basic_lstm_cell/kernel", "bidirectional_rnn/fw/basic_lstm_cell/bias",
                "bidirectional_rnn/bw/basic_lstm_cell/kernel", "bidirectional_rnn/bw/basic_lstm_cell/bias"]
}

# with torch.no_grad():   #eval
for _ in range(1):  # train
    enc = Encoder(32, 1, 256)
    mp = wt_map
    ones = torch.ones((4, 1, 32, 512))
    for i, layer in enumerate(enc.cnn):
        if str(i) in mp:
            if isinstance(layer, Conv2dSamePadding):
                enc.cnn[i].weight = torch.nn.Parameter(torch.tensor(d[mp[str(i)]]).permute(3, 2, 0, 1))
            elif isinstance(layer, BatchNorm2d):
                enc.cnn[i].weight = torch.nn.Parameter(torch.tensor(d[mp[str(i)][0]]).view(1, -1, 1, 1))
                enc.cnn[i].bias = torch.nn.Parameter(torch.tensor(d[mp[str(i)][1]]).view(1, -1, 1, 1))
                enc.cnn[i].moving_mean = torch.tensor(d[mp[str(i)][2]]).view(1, -1, 1, 1)
                enc.cnn[i].moving_variance = torch.tensor(d[mp[str(i)][3]]).view(1, -1, 1, 1)
        # ones = enc.cnn[i](ones)
    print(enc.cnn.state_dict().keys())
    enc.rnn.weight_ih_l0, enc.rnn.weight_hh_l0, enc.rnn.bias_ih_l0, enc.rnn.bias_hh_l0 = \
        get_pytorch_lstm_weights_from_tensorflow(d[wt_map["enc_rnn"][0]], d[wt_map["enc_rnn"][1]], 512)

    enc.rnn.weight_ih_l0_reverse, enc.rnn.weight_hh_l0_reverse, enc.rnn.bias_ih_l0_reverse, enc.rnn.bias_hh_l0_reverse = \
        get_pytorch_lstm_weights_from_tensorflow(d[wt_map["enc_rnn"][2]], d[wt_map["enc_rnn"][3]], 512)

    enc_outs, h = enc(ones)
    print("H",h.shape)
    # torch.save(enc.state_dict(),"models/encoder.pth")
    # print("ENC:", enc_outs)
    # print("HID:", h)
