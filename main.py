from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.attention_ocr import AttentionOcr, DataModule
from src.utils import helpers

if __name__ == "__main__":
    dm = DataModule(train_batch_size=2, val_batch_size=1, workers=4, img_h=32, img_w=512,
                    train_list='data/dataset/trainlist.txt')
    dm.prepare_data()
    dm.setup()

    model = AttentionOcr(32, 256, 2, helpers.get_alphabet())
    tr = Trainer(logger=TensorBoardLogger('logs/'), gpus=1)
    tr.fit(model, dm)
