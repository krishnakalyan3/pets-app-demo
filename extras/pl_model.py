import timm
import torch.nn as nn
import pytorch_lightning as pl


CLASSES = 120

class LitModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, CLASSES)
        )

    def forward(self, x):
        return self.model(x)

class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LitModel()
    
    def forward(self, x):
        x = self.model(x)
        return x