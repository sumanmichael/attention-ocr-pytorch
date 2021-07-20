from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from src.attention_ocr import AttentionOcr
from src.data import DataModule
from src.utils import helpers
from src.defaults import *
torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    dm = DataModule(train_batch_size=2, val_batch_size=2, workers=4, img_h=32, img_w=512,
                    train_list='data/dataset/train_list_uni.txt',val_list='data/dataset/train_list_uni.txt')
    dm.setup()

    h_params = {
        "lr": 1.0,
        "weight_decay": 0.95
    }

    model = AttentionOcr(IMAGE_HEIGHT,IMAGE_CHANNELS, ENC_HIDDEN_SIZE, ATTN_DEC_HIDDEN_SIZE, ENC_VEC_SIZE, ENC_SEQ_LENGTH, TARGET_EMBEDDING_SIZE, TARGET_VOCAB_SIZE, BATCH_SIZE, helpers.get_alphabet(), h_params)
    tr = Trainer(logger=TensorBoardLogger('logs/'), gradient_clip_val=5, gpus=1)
    # tr.tune(model, dm.train_dataloader())
    tr.fit(model, dm)
    # tr.validate(model, dm.val_dataloader())
