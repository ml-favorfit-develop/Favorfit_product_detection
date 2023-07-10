from imgaug import augmenters as iaa
from torchvision import transforms
import random

import torchvision.models.efficientnet


class ImgAugTransformer:
    resize = None
    resize_add = None
    resize_hue = None
    resize_add_hue = None
    normalize_to_tensor = None

    def __init__(self, height=224, width=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.resize_add = \
            iaa.Sequential([
                iaa.Resize({"height": height, "width": width}),
                iaa.Fliplr(0.5),
                iaa.Add((-40, 40)),
            ])

        self.resize_hue = \
            iaa.Sequential([
                iaa.Resize({"height": height, "width": width}),
                iaa.Fliplr(0.5),
                iaa.AddToHue((-20, 20))
            ])

        self.resize_add_hue = \
            iaa.Sequential([
                iaa.Resize({"height": height, "width": width}),
                iaa.Fliplr(0.5),
                iaa.Add((-40, 40)),
                iaa.AddToHue((-20, 20))
            ])

        self.normalize_to_tensor = \
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        self.resize = \
            iaa.Sequential([
                iaa.Resize({"height": height, "width": width}),
            ])

    def random_call(self, image):
        target_aug = random.choice([self.resize_add, self.resize_hue, self.resize_add_hue])
        x = target_aug(image=image)
        return self.normalize_to_tensor(x)

    def resize_call(self, image):
        x = self.resize(image=image)
        return self.normalize_to_tensor(x)

    def resize_add_call(self, image):
        x = self.resize_add(image=image)
        return self.normalize_to_tensor(x)

    def resize_hue_call(self, image):
        x = self.resize_hue(image=image)
        return self.normalize_to_tensor(x)

    def resize_add_hue_call(self, image):
        x = self.resize_add_hue(image=image)
        return self.normalize_to_tensor(x)

    def __call__(self, image, apply_aug=False):
        if apply_aug == True:
            return self.random_call(image)
        else:
            return self.resize_call(image)
