"""
Triplet-Center Loss implementation based on paper:
    He, X., Zhou, Y., Zhou, Z., Bai, S., & Bai, X. (2018). 
    Triplet-center loss for multi-view 3d object retrieval. 
    In Proceedings of the IEEE conference on computer vision 
    and pattern recognition (pp. 1945-1954).

Paper: 
    https://paperswithcode.com/paper/triplet-center-loss-for-multi-view-3d-object
"""

from torch.nn import Module, Parameter
from torch import normal, cdist, arange, relu


class TripletCenterLoss(Module):
    """
        Triplet Center Loss.

        Essa loss busca garantir que a distância entre o embedding de um exemplo e o centro
        da sua classe (distância positiva) seja menor do que a menor distância entre o embedding
        e os centros das demais classes (distância negativa) por pelo menos uma margem definida.

        # Parameters:
            num_classes (int): número de classes.
            feature_dim (int): dimensão do embedding.
            margin (float): margem para o triplet loss.
            device (str): .
    """
    def __init__(
            self, 
            num_classes: int, 
            feature_dim: int, 
            margin: float = 5.0, 
            mean_param: float = 0.0,
            std_param: float = 0.01,
            device='cpu'
    ):
        super(TripletCenterLoss, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.margin = margin
        self.centers = Parameter(
            normal(mean=mean_param, std=std_param, size=(num_classes, feature_dim)).to(device)
        )
    
    def forward(self, x, labels):
        """
            # Parameters::
                x: Matriz de features (embeddings) com shape (batch_size, feature_dim).
                labels: Rótulos verdadeiros com shape (batch_size).
            # Return:
                loss: valor da Triplet Center Loss.
        """
        
        batch_size = x.size(0)
        
        distances = cdist(x, self.centers, p=2).pow(2) 
        
        pos_distances = distances[arange(batch_size), labels]  
        
        mask = (
            arange(self.num_classes, device=x.device).unsqueeze(0) != labels.unsqueeze(1)
        )
        
        neg_distances = distances[mask].view(batch_size, -1).min(dim=1)[0]

        loss = relu(pos_distances - neg_distances + self.margin).mean()

        return loss
    