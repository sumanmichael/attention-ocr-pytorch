import random
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.modules.decoder_v2 import AttentionDecoder
from src.modules.encoder import Encoder
from src.utils import dataset
from src.utils import helpers


class AttentionOcr(pl.LightningModule):
    def __init__(self, img_h, n_hidden, batch_size, alphabet, h_params):
        super(AttentionOcr, self).__init__()

        self.h_params = h_params
        self.lr = self.h_params["lr"]
        self.n_class = len(alphabet) + 3
        self.encoder = Encoder(img_h, 1, n_hidden)
        self.decoder = AttentionDecoder(n_hidden, 128, self.n_class, dropout_p=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.converter = helpers.StrLabelConverterForAttention(alphabet)
        # Initialise Weights
        self.encoder.apply(helpers.weights_init)
        self.decoder.apply(helpers.weights_init)

        # TODO: Pretrained Model

        self.image_placeholder = torch.FloatTensor(batch_size, 3, img_h, img_h)

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):  # optimizer_idx is for dual_optimizer
        teach_forcing_prob = 0.5
        cpu_images, cpu_texts = train_batch
        b = cpu_images.size(0)
        target_variable = self.converter.encode(cpu_texts)
        helpers.loadData(self.image_placeholder, cpu_images)
        target_variable = target_variable.to(self.device)
        self.image_placeholder = self.image_placeholder.to(self.device)

        self.decoder.enc_outs = self.encoder(self.image_placeholder).permute_gates(1, 0, 2)
        #SOS
        decoder_y = target_variable[0].to('cpu')
        decoder_y = self.get_onehot(decoder_y, self.n_class)
        decoder_y = decoder_y.unsqueeze(1)
        decoder_hidden = self.decoder.init_hidden(b).to(self.device)
        decoder_out = decoder_y

        self.decoder.rnn_state = self.decoder.init_rnn_state()

        loss = 0.0
        for di in range(1, target_variable.shape[0]):  # Maximum string length
            decoder_y, decoder_out, decoder_hidden = self.decoder(
                    decoder_y, decoder_out, decoder_hidden)
            loss += self.criterion(decoder_y.squeeze(1), target_variable[di])

        # teach_forcing = True if random.random() > teach_forcing_prob else False
        # if teach_forcing:
        #     # Teacher Mandatory: Use the target label as the next input
        #     for di in range(1, target_variable.shape[0]):  # Maximum string length
        #         decoder_output, decoder_hidden, decoder_attention = self.decoder(
        #             decoder_input, decoder_hidden, encoder_outputs)
        #         loss += self.criterion(decoder_output, target_variable[di])  # Predict one character at a time
        #         decoder_input = target_variable[di]  # Teacher forcing/Previous output
        # else:
        #     for di in range(1, target_variable.shape[0]):
        #         decoder_output, decoder_hidden, decoder_attention = self.decoder(
        #             decoder_input, decoder_hidden, encoder_outputs)
        #         loss += self.criterion(decoder_output, target_variable[di])  # Predict one character at a time
        #         top_v, top_i = decoder_output.data.topk(1)
        #         ni = top_i.squeeze()
        #         decoder_input = ni

        self.log('train_loss', loss, logger=True)
        return loss

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


class DataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size, val_batch_size, workers, img_h, img_w, train_list=None, val_list=None,
                 test_list=None, keep_ratio=False, random_sampler=True):
        super(DataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.img_h = img_h
        self.img_w = img_w
        self.keep_ratio = keep_ratio
        self.random_sampler = random_sampler

        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        return

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_list:
            self.train_dataset = dataset.ListDataset(list_file=self.train_list)
        if self.val_list:
            self.val_dataset = dataset.ListDataset(list_file=self.val_list,
                                                   transform=dataset.ResizeNormalize((self.img_w, self.img_h)))
        if self.test_list:
            self.test_dataset = dataset.ListDataset(list_file=self.test_list)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            sampler=dataset.RandomSequentialSampler(
                self.train_dataset, self.train_batch_size) if self.random_sampler else None,
            num_workers=int(self.workers),
            collate_fn=dataset.AlignCollate(imgH=self.img_h, imgW=self.img_w, keep_ratio=self.keep_ratio))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.val_batch_size, num_workers=int(self.workers))
