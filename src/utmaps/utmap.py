from typing import Optional
from torch.nn import Module, Sequential, Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU


class ConvBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU()


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x


class UTMap(Module):
    
    def __init__(self, feature_dim: int, output_size: Optional[int] = None):
        super(UTMap, self).__init__()

        self.conv1 = ConvBlock(in_channels=512, out_channels=512)
        self.conv2 = ConvBlock(in_channels=512, out_channels=128)

        self.fc1 = Sequential(
            Linear(in_features=6_272, out_features=1024),
            BatchNorm1d(1024),
            ReLU(),
        )

        self.fc2 = Sequential(
            Linear(in_features=1024, out_features=128),
            BatchNorm1d(128),
            ReLU()
        )
        
        self.projection = Linear(in_features=128, out_features=feature_dim)

        self.has_classifier = False
        if output_size is not None:
            self.has_classifier = True
            self.classifier = Linear(in_features=128, out_features=output_size)


    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)

        outputs = outputs.view(outputs.size(0), -1)

        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)

        embeddings = self.projection(outputs)

        if self.has_classifier:
            logits = self.classifier(outputs)
            return embeddings, logits

        return embeddings
    

def load_utmap(model_state: dict) -> UTMap:
    
    feature_dim = model_state['projection.weight'].shape[0]

    model_keys = list(model_state.keys())

    output_size = model_state['classifier.weight'].shape[0] if 'classifier.weight' in model_keys else None 
    
    trimap = UTMap(
        feature_dim=feature_dim,
        output_size=output_size
    )

    trimap.load_state_dict(model_state)

    return trimap