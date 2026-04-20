import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_model(num_classes=2, fine_tune=True):
    # ConvNext Tiny modelini yükle
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    if fine_tune:
        # DEĞİŞİKLİK BURADA: Sadece features.7 değil, features.6'yı da eğitime açıyoruz.
        for name, param in model.named_parameters():
            if "features.5" in name or "features.6" in name or "features.7" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # Tüm katmanları dondur (sadece sınıflandırıcı hariç)
        for param in model.parameters():
            param.requires_grad = False
    
    # Sınıflandırıcı katmanı (classifier) her zaman eğitilebilir kalır
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.6), # Ezberlemeyi önlemek için dropout değeri yüksek tutulmuştur
        nn.Linear(128, num_classes)
    )
    return model

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x