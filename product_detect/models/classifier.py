import torch.nn as nn
from torchvision.models import *

class Classifier(nn.Module):
    def __init__(self, base_model, num_classes=2, **kwargs):
        super(Classifier, self).__init__()

        if base_model == 'resnet18':
            self.base_model = resnet18(**kwargs)
        elif base_model == 'resnet101':
            self.base_model = resnet101(**kwargs)
        elif base_model == 'efficientnet_b0':
            self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1, **kwargs)
        elif base_model == 'efficientnet_b4':
            self.base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1, **kwargs)
        elif base_model == 'efficientnet_b7':
            self.base_model = efficientnet_b7(**kwargs)
        else:
            print('Model name {} is not implemented yet!'.format(base_model))
            raise TypeError
        
        out_features = self.base_model.classifier[-1].out_features
        self.fc = nn.Linear(out_features, num_classes)
    
    def forward(self, x):
        output = self.base_model(x)
        output = self.fc(output)
        return output