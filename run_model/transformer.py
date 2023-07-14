from torchvision import transforms

class ImgResizeTransformer:

    def __init__(self, height=224, width=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.normalize_to_tensor = \
            transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def resize_call(self, image):
        return self.normalize_to_tensor(image)

    def __call__(self, image):
        return self.resize_call(image)
