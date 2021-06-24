from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.attention_ocr import AttentionOcr, DataModule
from src.utils import helpers

if __name__ == "__main__":
    dm = DataModule(train_batch_size=2, val_batch_size=1, workers=4, img_h=32, img_w=512,
                    train_list='data/dataset/trainlist.txt')
    dm.setup()

    h_params = {
        "lr": 1.0,
        "weight_decay": 0.95
    }

    model = AttentionOcr(32, 256, 2, helpers.get_alphabet(), h_params)
    tr = Trainer(logger=TensorBoardLogger('logs/'), gpus=1, gradient_clip_val=0.5)
    # tr.tune(model, dm.train_dataloader())
    tr.fit(model, dm)
