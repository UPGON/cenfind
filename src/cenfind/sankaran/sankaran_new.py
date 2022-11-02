import albumentations as alb
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cenfind.core.data import Dataset as Cenfind_dataset
from cenfind.sankaran.datasets import FociDatasetSankaran
from cenfind.sankaran.helpers import (compute_focal_loss_weights)
from cenfind.sankaran.models import MultiChannelCombinedScorer

transforms = alb.Compose([
    alb.ShiftScaleRotate(scale_limit=0.),
    alb.Flip(),
])


class LitDetector(pl.LightningModule):
    def __init__(self, model, gamma=None, need_sigmoid=False):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.need_sigmoid = need_sigmoid

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x)
        y_hat = F.interpolate(y_hat, size=y.shape[-2:])
        if self.gamma is not None:
            weight = compute_focal_loss_weights(y_hat, y, self.gamma)
            w = weight * w
            w = w.detach().squeeze(0)
        if self.need_sigmoid:
            loss = F.binary_cross_entropy_with_logits(y_hat[:, 0, :, :], y, weight=w)
        else:
            loss = F.binary_cross_entropy(y_hat[:, 0, :, :], y, weight=w)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=.9, weight_decay=1e-4)
        return optimizer


def main():
    ds = Cenfind_dataset('/data1/centrioles/RPE1p53+Cnone_CEP63+CETN2+PCNT_1/')
    fixed_channel = 1

    train_dataset = FociDatasetSankaran(ds, 'train', channel=fixed_channel)
    train_loader = DataLoader(train_dataset)

    test_dataset = FociDatasetSankaran(ds, 'test', channel=fixed_channel)
    test_loader = DataLoader(test_dataset)

    detector = LitDetector(MultiChannelCombinedScorer(), gamma=2)

    trainer = pl.Trainer(accelerator='gpu', devices=2, max_epochs=100, log_every_n_steps=5)

    trainer.fit(model=detector, train_dataloaders=train_loader)
    trainer.test(model=detector, dataloaders=test_loader)


if __name__ == '__main__':
    main()
