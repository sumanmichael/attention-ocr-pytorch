import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.modules.decoder import AttentionDecoder
from src.modules.encoder import Encoder
from src.utils import helpers
from src.defaults import *


class AttentionOcr(pl.LightningModule):
    def __init__(self, image_height, image_channels, enc_hidden_size,attn_dec_hidden_size, enc_vec_size, enc_seq_length, target_embedding_size, target_vocab_size,
                 batch_size, alphabet, h_params):
        super(AttentionOcr, self).__init__()

        self.h_params = h_params
        self.lr = self.h_params["lr"]

        self.n_class = len(alphabet) #+ 3
        self.encoder = Encoder(image_height, image_channels, enc_hidden_size)
        self.decoder = AttentionDecoder(attn_dec_hidden_size, enc_vec_size, enc_seq_length, target_embedding_size, target_vocab_size,batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.converter = helpers.StrLabelConverterForAttention(alphabet)
        # Initialise Weights
        # self.encoder.apply(helpers.weights_init)
        # self.decoder.apply(helpers.weights_init)

        # TODO: Pretrained Model

        self.image_placeholder = torch.FloatTensor(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

    def training_step(self, train_batch, batch_idx):  # optimizer_idx is for dual_optimizer
        attention_context, state, target_variable = self.run_enc(train_batch)

        loss = 0.0
        for di in range(0, target_variable.shape[0]):  # Maximum string length
            prev_output = self.get_onehot(target_variable[di], self.n_class)
            prev_output, attention_context, state = self.decoder(prev_output, attention_context, state)
            loss += self.criterion(prev_output.squeeze(1), target_variable[di])

        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, train_batch, batch_idx):
        attention_context, state, target_variable = self.run_enc(train_batch)

        prev_output = self.get_onehot(torch.tensor([1]*BATCH_SIZE), self.n_class)
        loss = 0.0
        for di in range(0, target_variable.shape[0]):  # Maximum string length
            prev_output, attention_context, state = self.decoder(prev_output, attention_context, state)
            loss += self.criterion(prev_output.squeeze(1), target_variable[di])

        self.log('val_loss', loss, logger=True)
        return loss

    def run_enc(self, train_batch):
        cpu_images, cpu_texts = train_batch
        target_variable = self.converter.encode(cpu_texts)
        helpers.loadData(self.image_placeholder, cpu_images)
        target_variable = target_variable.to(self.device)
        self.image_placeholder = self.image_placeholder.to(self.device)
        self.image_placeholder = torch.ones_like(self.image_placeholder)
        enc_out, state = self.encoder(self.image_placeholder)
        self.decoder.set_encoder_output(enc_out)
        state = helpers.modify_state_for_tf_compat(state)  # .to(self.device)
        state = state[0].to(self.device), state[1].to(self.device)
        attention_context = torch.zeros((BATCH_SIZE, ENC_VEC_SIZE)).to(self.device)
        return attention_context, state, target_variable

    def get_onehot(self, arr, max_value):
        arr = arr.to('cpu')
        return torch.zeros(len(arr), max_value).scatter_(1, arr.unsqueeze(1), 1.).to(self.device)

    # def configure_optimizers(self):
    #     encoder_optimizer = torch.optim.Adam(self.encoder.parameters())
    #     decoder_optimizer = torch.optim.Adam(self.decoder.parameters())
    #     return encoder_optimizer, decoder_optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr, weight_decay=self.h_params["weight_decay"])
        return optimizer


