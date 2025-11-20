# -*- coding: utf-8 -*-


import torch
from torch.utils.data import Dataset
from model.TorViNet import TorViNet
from trainer import Trainer


# --------------------------------------------------
# Replace with your own dataset
# --------------------------------------------------
class VideoDataset(Dataset):
    """
    example dataset：output (video_tensor, label)
    Replace with your own dataset
    """
    pass


if __name__ == "__main__":

    # --------------------------------------------------
    # load dataset
    # --------------------------------------------------
    train_dataset = VideoDataset()
    val_dataset = VideoDataset()

    # --------------------------------------------------
    # Initialize the model
    # --------------------------------------------------
    model = TorViNet()

    # --------------------------------------------------
    # Initialize Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        lr=0.001,               # Init lr
        min_lr=1e-5,            # cosine annealing
        num_epochs=50,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------
    # Start train
    # --------------------------------------------------
    trainer.train()

