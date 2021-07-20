import torch

from src.modules.encoder import Encoder
from src.modules.decoder import AttentionDecoder
from port_enc import get_pytorch_lstm_weights_from_tensorflow
import pickle

# torch.set_default_dtype(torch.float64)
from src.utils.helpers import get_one_hot

d = pickle.load(open(r'C:\Users\suman\iiith\attention-ocr-pytorch\wts.pkl', "rb"))
# for k in sorted(d.keys()):
#     print(k, d[k].shape)

enc = Encoder(32, 1, 256)
dec = AttentionDecoder(128, 512, 128, 10, 270, 4)


def get_wt(s, perm=None, unsqueeze=None):
    tensor = torch.tensor(d[s])
    if perm is not None:
        tensor = tensor.permute(*perm)
    if unsqueeze is not None:
        tensor = tensor.unsqueeze(unsqueeze)
    return tensor


wt_data = {
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
    'VT.weight':{
        'wt': 'embedding_attention_decoder/attention_decoder/AttnV_0',
        'unsqueeze': 0
    }
}

state_dict = {}
for k,v in wt_data.items():
    state_dict[k] = get_wt(v["wt"],v["perm"] if "perm" in v else None, v["unsqueeze"] if "unsqueeze" in v else None)
print(state_dict)
dec_dict = dec.state_dict()
dec_dict.update(state_dict)
dec.load_state_dict(dec_dict)

w_ih, w_hh, b_ih, b_hh = get_pytorch_lstm_weights_from_tensorflow(
    d['embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'],
    d['embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/basic_lstm_cell/bias'],
    INPUT_SIZE=128
)

dec.rnn.weight_ih_l0, dec.rnn.weight_hh_l0, dec.rnn.bias_ih_l0, dec.rnn.bias_hh_l0 = w_ih, w_hh, b_ih, b_hh
dec.rnn.weight_ih_l1, dec.rnn.weight_hh_l1, dec.rnn.bias_ih_l1, dec.rnn.bias_hh_l1 = w_ih, w_hh, b_ih, b_hh

torch.save(dec.state_dict(),r'C:\Users\suman\iiith\attention-ocr-pytorch\models\decoder.pth')


