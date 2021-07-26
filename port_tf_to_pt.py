import torch
from torch import nn

from src.modules.decoder import AttentionDecoder
from src.modules.encoder import Encoder
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
    b_hh = nn.Parameter(torch.tensor(bih) * (1 - alpha))

    return w_ih, w_hh, b_ih, b_hh


def get_wt(s, perm=None, unsqueeze=None, view=None):
    tensor = torch.tensor(d[s])
    if perm is not None:
        tensor = tensor.permute(*perm)
    if unsqueeze is not None:
        tensor = tensor.unsqueeze(unsqueeze)
    if view is not None:
        tensor = tensor.view(*view)
    return tensor


def load_enc_weights(enc, d):
    enc_cnn = {
        "cnn.0.weight": {
            "wt": 'conv_conv1/W',
            "perm": (3, 2, 0, 1)
        },
        "cnn.3.weight": {
            "wt": 'conv_conv2/W',
            "perm": (3, 2, 0, 1)
        },
        "cnn.6.weight": {
            "wt": 'conv_conv3/W',
            "perm": (3, 2, 0, 1)
        },
        "cnn.9.weight": {
            "wt": 'conv_conv4/W',
            "perm": (3, 2, 0, 1)
        },
        "cnn.12.weight": {
            "wt": 'conv_conv5/W',
            "perm": (3, 2, 0, 1)
        },

        "cnn.15.weight": {
            "wt": 'conv_conv6/W',
            "perm": (3, 2, 0, 1)
        },
        "cnn.18.weight": {
            "wt": 'conv_conv7/W',
            "perm": (3, 2, 0, 1)
        },

        "cnn.7.weight": {
            "wt": 'conv_conv3/BatchNorm/gamma',
            "view": (1, -1, 1, 1)
        },

        "cnn.7.bias": {
            "wt": 'conv_conv3/BatchNorm/beta',
            "view": (1, -1, 1, 1)
        },

        "cnn.13.weight": {
            "wt": 'conv_conv5/BatchNorm/gamma',
            "view": (1, -1, 1, 1)
        },

        "cnn.13.bias": {
            "wt": 'conv_conv5/BatchNorm/beta',
            "view": (1, -1, 1, 1)
        },

        "cnn.19.weight": {
            "wt": 'conv_conv7/BatchNorm/gamma',
            "view": (1, -1, 1, 1)
        },

        "cnn.19.bias": {
            "wt": 'conv_conv7/BatchNorm/beta',
            "view": (1, -1, 1, 1)
        },
    }

    pre_enc_rnn = {
        "l0": "bidirectional_rnn/fw/basic_lstm_cell/",
        "l0_reverse": "bidirectional_rnn/fw/basic_lstm_cell/"
    }



    enc_bn = {
        "7.moving_mean": {
            "wt": 'conv_conv3/BatchNorm/moving_mean',
            "view": (1, -1, 1, 1)
        },

        "7.moving_variance": {
            "wt": 'conv_conv3/BatchNorm/moving_variance',
            "view": (1, -1, 1, 1)
        },

        "13.moving_mean": {
            "wt": 'conv_conv5/BatchNorm/moving_mean',
            "view": (1, -1, 1, 1)
        },

        "13.moving_variance": {
            "wt": 'conv_conv5/BatchNorm/moving_variance',
            "view": (1, -1, 1, 1)
        },

        "19.moving_mean": {
            "wt": 'conv_conv7/BatchNorm/moving_mean',
            "view": (1, -1, 1, 1)
        },

        "19.moving_variance": {
            "wt": 'conv_conv7/BatchNorm/moving_variance',
            "view": (1, -1, 1, 1)
        },
    }

    enc_state_dict = {}
    for k, v in enc_cnn.items():
        enc_state_dict[k] = get_wt(
            v["wt"],
            v["perm"] if "perm" in v else None,
            v["unsqueeze"] if "unsqueeze" in v else None,
            v["view"] if "view" in v else None
        )

    for k, v in pre_enc_rnn.items():
        w_ih, w_hh, b_ih, b_hh = get_pytorch_lstm_weights_from_tensorflow(
            d[v + "kernel"],
            d[v + "bias"],
            INPUT_SIZE=512
        )
        enc_state_dict["rnn.weight_ih_" + k] = w_ih
        enc_state_dict["rnn.weight_hh_" + k] = w_hh
        enc_state_dict["rnn.bias_ih_" + k] = b_ih
        enc_state_dict["rnn.bias_hh_" + k] = b_hh

    enc_dict = enc.state_dict()
    enc_dict.update(enc_state_dict)
    enc.load_state_dict(enc_dict)

    for i in [7, 13, 19]:
        enc.cnn[i].moving_mean = get_wt(
            enc_bn[f'{i}.moving_mean']["wt"],
            view=enc_bn[f'{i}.moving_mean']["view"]
        )
        enc.cnn[i].moving_mean = get_wt(
            enc_bn[f'{i}.moving_variance']["wt"],
            view=enc_bn[f'{i}.moving_variance']["view"]
        )


def load_dec_weights(dec, d):
    dec_wt_map = {
        'conv_1x1.weight': {
            'wt': 'embedding_attention_decoder/attention_decoder/AttnW_0',
            'perm': (3, 2, 0, 1)
        },
        'embedding.weight': {
            'wt': 'embedding_attention_decoder/embedding'
        },
        'attention_projection.weight': {
            'wt': 'embedding_attention_decoder/attention_decoder/Attention_0/kernel',
            'perm': (1, 0)
        },
        'attention_projection.bias': {
            'wt': 'embedding_attention_decoder/attention_decoder/Attention_0/bias'
        },
        'input_projection.weight': {
            'wt': 'embedding_attention_decoder/attention_decoder/kernel',
            'perm': (1, 0)
        },
        'input_projection.bias': {
            'wt': 'embedding_attention_decoder/attention_decoder/bias'
        },
        'output_projection.weight': {
            'wt': 'embedding_attention_decoder/attention_decoder/AttnOutputProjection/kernel',
            'perm': (1, 0)
        },
        'output_projection.bias': {
            'wt': 'embedding_attention_decoder/attention_decoder/AttnOutputProjection/bias'
        },
        'VT.weight': {
            'wt': 'embedding_attention_decoder/attention_decoder/AttnV_0',
            'unsqueeze': 0
        }
    }

    dec_state_dict = {}
    for k, v in dec_wt_map.items():
        dec_state_dict[k] = get_wt(v["wt"], v["perm"] if "perm" in v else None,
                                   v["unsqueeze"] if "unsqueeze" in v else None)
    dec_dict = dec.state_dict()
    dec_dict.update(dec_state_dict)
    dec.load_state_dict(dec_dict)

    w_ih, w_hh, b_ih, b_hh = get_pytorch_lstm_weights_from_tensorflow(
        d['embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'],
        d['embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/basic_lstm_cell/bias'],
        INPUT_SIZE=128
    )

    dec.rnn.weight_ih_l0, dec.rnn.weight_hh_l0, dec.rnn.bias_ih_l0, dec.rnn.bias_hh_l0 = w_ih, w_hh, b_ih, b_hh
    dec.rnn.weight_ih_l1, dec.rnn.weight_hh_l1, dec.rnn.bias_ih_l1, dec.rnn.bias_hh_l1 = w_ih, w_hh, b_ih, b_hh


if __name__ == "__main__":
    d = pickle.load(open("wts.pkl", "rb"))
    enc = Encoder(32, 1, 256)
    dec = AttentionDecoder(128, 512, 128, 10, 270, 4)

    load_enc_weights(enc, d)
    load_dec_weights(dec, d)

    print(enc(torch.ones(1, 1, 32, 512)))
