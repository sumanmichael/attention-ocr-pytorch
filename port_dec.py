import torch

from src.modules.encoder import Encoder
from src.modules.decoder import AttentionDecoder
from port_tf_to_pt import get_pytorch_lstm_weights_from_tensorflow
import pickle

# torch.set_default_dtype(torch.float64)
from src.utils.helpers import get_one_hot

d = pickle.load(open(r'C:\Users\suman\iiith\attention-ocr-pytorch\wts.pkl', "rb"))
# for k in sorted(d.keys()):
#     print(k, d[k].shape)

enc = Encoder(32, 1, 256)
dec = AttentionDecoder(128, 512, 128, 10, 270, 4)




torch.save(dec.state_dict(),r'C:\Users\suman\iiith\attention-ocr-pytorch\models\decoder.pth')


