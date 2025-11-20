# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=64,
        lr=1e-3,
        min_lr=1e-5,
        num_epochs=50,
        num_workers=4,
        device="cuda"
    ):
        self.model = model.to(device)
        self.device = device

        # -----------------------------
        # DataLoader
        # -----------------------------
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

        # -----------------------------
        # Loss & Optimizer & Scheduler
        # -----------------------------
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=min_lr
        )

        self.num_epochs = num_epochs

    # -----------------------------
    # Xavier Initialization
    # -----------------------------
    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", ncols=100)

        for batch in pbar:
            videos, labels = batch
            videos = videos.to(self.device)      # [B, 3, 64, 224, 224]
            labels = labels.to(self.device)      # [B]

            preds = self.model(videos)           # output [B, num_classes]
            loss = self.criterion(preds, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(videos)
                loss = self.criterion(preds, labels)
                total_loss += loss.item()

                _, predicted = torch.max(preds, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        return total_loss / len(self.val_loader), acc

    def train(self):

        self.init_weights()
        print("=> Model initialized with Xavier initialization.")

        for epoch in range(1, self.num_epochs + 1):

            train_loss = self.train_one_epoch(epoch)

            if self.val_loader is not None:
                val_loss, val_acc = self.validate()
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

            self.scheduler.step()
