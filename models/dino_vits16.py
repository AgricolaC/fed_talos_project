import torch
import torch.nn as nn


class DINO_ViT(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, frozen_backbone=False, head_dropout=0.25):
        super(DINO_ViT, self).__init__()
        
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)

        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(384, num_classes)  # DINO ViT-S/16 output dim
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

            