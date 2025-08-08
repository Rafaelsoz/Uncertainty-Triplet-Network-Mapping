"""
ResNet in PyTorch implementation based on:
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Paper:
    https://arxiv.org/abs/1512.03385
"""

from typing import Callable, Optional

from tqdm import tqdm
from torch import no_grad, where, cat
from torch.utils.data import Dataset, DataLoader

from torch.nn import Linear, Module, Sequential, Flatten
from torchvision.models import resnet18, ResNet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode


def get_resnet18(
        output_size: int,
        freeze_layers: bool = False,
) -> ResNet:
    
    model = resnet18(weights="IMAGENET1K_V1")
    
    for param in model.parameters():
        param.requires_grad = not freeze_layers
        
    num_in_features = model.fc.in_features
    model.fc = Linear(num_in_features, output_size)

    return model


def get_resenet18_transform(custom_augmentation: Optional[Callable] = None):
    center_crop_value = 224
    resize_value = 256

    if custom_augmentation is None:
       transform = Compose([
            Resize([resize_value], interpolation=InterpolationMode.BILINEAR),
            CenterCrop(center_crop_value),
            ToTensor()
        ])
       
    else:
        transform = Compose([
            Resize([resize_value], interpolation=InterpolationMode.BILINEAR),
            CenterCrop(center_crop_value),
            custom_augmentation,
            ToTensor()
        ])

    return transform


def load_resnet18(model_state: dict, freeze_layers: bool = True) -> ResNet:

    last_layer_key = list(model_state.keys())[-1]
    output_size = model_state[last_layer_key].shape[0]

    resnet = get_resnet18(output_size=output_size, freeze_layers=freeze_layers)

    resnet.load_state_dict(model_state)

    return resnet


class BackboneResNet18(Module):
    def __init__(self, model_state: dict, requires_grad: bool = False):
        super().__init__()
        
        last_layer_key = list(model_state.keys())[-1]
        output_size = model_state[last_layer_key].shape[0]

        model = get_resnet18(output_size=output_size, freeze_layers=True)
        model.load_state_dict(model_state)

        for param in model.parameters():
            param.requires_grad = requires_grad

        self.conv_layers = Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        self.avgpool = model.avgpool
        self.flatten = Flatten()
        self.fc = model.fc


    def forward(self, x):
        outputs = self.conv_layers(x)
        outputs = self.avgpool(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc(outputs)
        return outputs
    

    def get_embeddings(self, x):
        outputs = self.conv_layers(x)
        outputs = self.avgpool(outputs)
        outputs = self.flatten(outputs)
        return outputs
    

    def get_embeddings_and_logits(self, x):
        embeddings = self.get_embeddings(x)
        logits = self.fc(embeddings)
        return embeddings, logits
    

    def get_featuremaps(self, x):
        outputs = self.conv_layers(x)
        return outputs
    

    def get_featuremaps_and_logits(self, x):
        featuremaps = self.conv_layers(x)
        logits = self.fc(self.flatten(self.avgpool(featuremaps)))
        return featuremaps, logits
    

    def get_featuremaps_embeddings_and_logits(self, x):
        featuremaps = self.conv_layers(x)
        embeddings = self.avgpool(featuremaps)
        embeddings = self.flatten(embeddings)
        logits = self.fc(embeddings)
        return featuremaps, embeddings, logits
    

def get_backbone_predicts(
    num_classes: int,
    backbone: BackboneResNet18,
    dataset: Dataset,
    device: str = 'cpu',
    th: float = 0.5
):
    backbone.eval()
    backbone.to(device)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    predicts = []
    targets = []

    with no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.squeeze() if labels.dim() > 1 else labels

            logits = backbone(inputs)

            if num_classes > 2:
                probs = logits.softmax(dim=1)
                preds = probs.argmax(dim=1).squeeze()
            else:
                probs = logits.sigmoid().squeeze()
                preds = where(probs < th, 0, 1)

            predicts.append(preds.cpu())
            targets.append(labels.cpu())
            

    predicts = cat(predicts)
    targets = cat(targets)

    return targets, predicts