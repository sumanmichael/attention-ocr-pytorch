import torch

from src.defaults import *
from src.modules.decoder import AttentionDecoder
from src.modules.encoder import Encoder
from src.utils.helpers import get_one_hot, modify_state_for_tf_compat

enc = Encoder(IMAGE_WIDTH, IMAGE_CHANNELS, ENC_HIDDEN_SIZE)
enc_out, state = enc(torch.ones(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT))
state = modify_state_for_tf_compat(state)

assert ENC_SEQ_LENGTH == enc_out.shape[0]
assert ENC_VEC_SIZE == enc_out.shape[2]

dec = AttentionDecoder(ATTN_DEC_HIDDEN_SIZE, ENC_VEC_SIZE, ENC_SEQ_LENGTH, TARGET_EMBEDDING_SIZE, TARGET_VOCAB_SIZE,
                       BATCH_SIZE)

dec.set_encoder_output(enc_out)

prev_output = get_one_hot([1] * BATCH_SIZE)
attention_context = torch.zeros((BATCH_SIZE, ENC_VEC_SIZE))

with torch.no_grad():
    for i in range(2):
        prev_output, attention_context, state = dec(prev_output, attention_context, state)
        print(prev_output)
        prev_output = get_one_hot([2] * BATCH_SIZE)