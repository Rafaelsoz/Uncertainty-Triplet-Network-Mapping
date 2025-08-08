from torch.nn import Module
from torch.nn.functional import pairwise_distance, relu

class TripletLoss(Module):

    def __init__(self, margin: float):
        super(TripletLoss, self).__init__()

        self.margin = margin
    
    def forward(self, anchor, positive, negative):

        positive_distance = pairwise_distance(anchor, positive, p=2)
        negative_distance = pairwise_distance(anchor, negative, p=2)

        loss = relu(positive_distance - negative_distance + self.margin)

        return loss.mean()