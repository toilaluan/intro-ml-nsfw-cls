from pl_text_module import TextClassification
from data import TextDataset
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', type=bool, default=False)
    parser.add_argument("--root_path", type=str, default="dataset")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pretrain_name", type=str, default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--weight_decay", default=1e-3)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    L = TextClassification(args.pretrain_name, args.max_length, args)

    train_dataset = TextDataset(args.root_path, True)
    val_dataset = TextDataset(args.root_path, False)

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    logger = TensorBoardLogger("train_logs", name="full_train", version=args.pretrain_name)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=10,
        devices=[args.device],
    )
    trainer.fit(
        model=L, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
