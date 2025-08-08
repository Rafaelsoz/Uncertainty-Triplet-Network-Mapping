"""
ResNet in PyTorch implementation based on:
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Paper:
    https://arxiv.org/abs/1512.03385
"""

from tqdm import tqdm
from torch import where, cat, log, softmax, tensor
from torch.nn.functional import relu
from torch.nn import Dropout2d
from torchvision.models.resnet import ResNet, BasicBlock
from torch import flatten, no_grad, stack
from torch.utils.data import Dataset, DataLoader


class DropoutBasicBlock(BasicBlock):
    def __init__(self, *args, p: float=0.1, **kwargs):
        super(DropoutBasicBlock, self).__init__(*args, **kwargs)
        self.dropout = Dropout2d(p=p)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        # out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = relu(out)
        out = self.dropout(out)

        return out

class ResNet18Dropout(ResNet):
    def __init__(self, num_classes: int, p: float=0.1):
        super(ResNet18Dropout, self).__init__(
            block=DropoutBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=num_classes
        )
        
        self.multiclass_task = num_classes > 2

        for module in self.modules():
            if isinstance(module, DropoutBasicBlock):
                module.p = p
        
        self.dropout = Dropout2d(p=p)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)
        
        return x

    def mc_forward(self, x, num_samples: int=50):
        with no_grad():
            if self.multiclass_task:
                outputs = [softmax(self.forward(x), dim=1) for _ in range(num_samples)]
            else:
                outputs = [self.forward(x).sigmoid() for _ in range(num_samples)]
        return stack(outputs)
    

def load_resnet18_dropout(model_state: dict, freeze_layers: bool = False) -> ResNet:

    last_layer_key = list(model_state.keys())[-1]
    output_size = model_state[last_layer_key].shape[0]

    resnet = ResNet18Dropout(num_classes=output_size)

    resnet.load_state_dict(model_state)

    for param in resnet.parameters():
        param.requires_grad = not freeze_layers

    return resnet


def get_resnet_dropout_predicts(
    num_classes: int,
    classifier: ResNet18Dropout,
    dataset: Dataset,
    num_samples: int=50,
    device: str = 'cpu',
    th: float = 0.5
):
    classifier.train()
    classifier.to(device)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    h_max = log(tensor(num_classes))

    predicts = []
    targets = []
    entropys = []

    with no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.squeeze() if labels.dim() > 1 else labels

            if num_classes > 2:
                mc_probs = classifier.mc_forward(inputs, num_samples=num_samples)
                mean_probs = mc_probs.mean(dim=0)
                preds = mean_probs.argmax(dim=1).squeeze()
                probs = mean_probs
            else:
                mc_probs = classifier.mc_forward(inputs, num_samples=num_samples)
                mean_probs = mc_probs.mean(dim=0)
                preds = where(mean_probs.squeeze() < th, 0, 1)
                probs = cat([1 - mean_probs, mean_probs], dim=1)
            
            norm_entropy = (probs * log(probs + 1e-9)).sum(dim=1) / h_max
            norm_entropy = norm_entropy.squeeze() if norm_entropy.dim() > 1 else norm_entropy

            predicts.append(preds.cpu())
            targets.append(labels.cpu())
            entropys.append(norm_entropy.cpu()) 

    predicts = cat(predicts)
    targets = cat(targets)
    entropys = cat(entropys)

    return targets, predicts, entropys