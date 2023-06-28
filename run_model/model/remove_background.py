from transparent_background import Remover
import torch.nn as nn

class BackGroundRemover(nn.Module):
    def __init__(self, base_model, **kwargs):
        super(BackGroundRemover, self).__init__()

        if base_model == "inspyrenet":
            self.base_model_name = base_model
            self.model = Remover(**kwargs).process
        else:
            print(f"Model name {base_model} is not implemented yet!")
            raise TypeError
    
    def forward(self, x):
        output = self.model(x, type='map')
        return output
    