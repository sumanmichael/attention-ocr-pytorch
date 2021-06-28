from typing import Optional

import pytorch_lightning as pl
import torch

from src.utils import dataset


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