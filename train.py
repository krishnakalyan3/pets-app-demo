#!/usr/bin/env python3

import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, MetricCollection
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
from PIL import Image
import timm
import wandb
import numpy as np
from datetime import datetime


current_time = datetime.now().strftime("%D %H:%M:%S"); current_time
wandb.init(project="pets")
config = wandb.config
print(config)

def is_valid(files, target_dict, split_pct = .2):
    split_class = {j:0 for j in target_dict}
    train_idx = []
    val_idx = []
    
    # Calculate number of images perclass
    for i in files:
        class_name = i.parent.name
        split_class[class_name] += 1
    
    # Calculate percentage
    for i in split_class:
        split_class[i] = int(split_class[i]*split_pct)
    
    for idx, i in enumerate(files):
        class_name = i.parent.name
        
        if split_class[class_name] == 0:
            train_idx.append(idx)
        else:
            val_idx.append(idx)
            split_class[class_name] -= 1
    
    return train_idx, val_idx

aug = A.Compose([
            A.Resize(225, 225),
            A.HorizontalFlip(0.5),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Rotate(10),
            A.ColorJitter(0.2,0.2,0,0),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ], p=1.0)

class Pets(Dataset):
    def __init__(self, 
                 data_dir: str = None,
                 transforms = T.Compose([T.Resize((225, 225)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
        super().__init__()
        self.files = [i for i in data_dir.glob("*/*.jpg")]
        self.transforms = transforms
        self.target_dict = {k.name:i for i,k in enumerate(data_dir.iterdir())}
        self.inverse_target = {i:k.name for i,k in enumerate(data_dir.iterdir())}

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path  = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transforms:
            img = np.array(img)
            img = self.transforms(image=img)['image']

        return {'image':img, 'target': self.target_dict[img_path.parent.name]}
    
ROOT = Path("images/Images/")
dset = Pets(data_dir=ROOT, transforms=aug)
classes = len(dset.target_dict)
class_names = list(dset.target_dict.keys())
index = is_valid(dset.files, dset.target_dict)

class PetsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, index, dset):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        (self.train_idx, self.val_idx) = index
        self.dset = dset

    def train_dataloader(self):
        train_dset = Subset(dset, self.train_idx)
        return DataLoader(train_dset, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        val_dset = Subset(dset, self.val_idx)
        return DataLoader(val_dset, batch_size=self.batch_size, num_workers=4)

class LitModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    
class LitClassifier(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = MetricCollection({"train_acc": Accuracy(num_classes=classes, average="micro", multiclass=True),
                                               "train_f1": F1Score(num_classes=classes, average="macro", multiclass=True)})
        self.val_metrics = MetricCollection({"val_acc": Accuracy(num_classes=classes, average="micro", multiclass=True),
                                             "val_f1": F1Score(num_classes=classes, average="macro", multiclass=True)
                                            })
        
        self.learning_rate = learning_rate
        self.model = LitModel()
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = batch['image']
        y = batch['target']
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)

        logs = {'train_loss': loss, 
                'train_accuracy': self.train_metrics["train_acc"], 
                "train_f1": self.train_metrics["train_f1"]}
        wandb.log(logs)
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = batch['image']
        y = batch['target']
        y_hat = self.model(x)        
        loss = self.criterion(y_hat, y)

        logs = {'val_loss': loss, 
                'val_accuracy': self.val_metrics["val_acc"], 
                "val_f1": self.val_metrics["val_f1"]}
        wandb.log(logs)
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"}}


def main():
    model = LitClassifier(learning_rate=0.001)
    pets = PetsDataModule(ROOT, config.batch_size, index, dset)
    trainer = pl.Trainer(max_epochs=config.epochs, 
                     accelerator='auto',
                     devices=1, 
                     precision=16,
                     enable_progress_bar=True,
                     callbacks=[RichProgressBar()])
    trainer.fit(model, pets)

main()
