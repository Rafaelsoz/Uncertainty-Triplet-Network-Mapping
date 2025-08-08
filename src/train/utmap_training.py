import numpy as np

from tqdm import trange

from argparse import Namespace, ArgumentParser

from typing import Optional, Callable

from torch import no_grad, Generator, where
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn.utils import clip_grad_value_

from ..utmaps import UTMap
from ..classifiers.resnet18 import BackboneResNet18
from ..losses.triplet_center_loss import TripletCenterLoss
from ..losses.triplet_center_softmax_loss import TripletCenterSoftmaxLoss
from ..utils.model_checkpoint import Checkpoint
from ..utils.metrics_logger import MetricsLogger
from ..utils.label_mapping import create_modified_labels


def create_utmap_argparse(
    model_name: Optional[str],
    num_classes: Optional[int],
    loss: Optional[str],
    feature_dim: Optional[int],
    epochs: Optional[int],
    batch_size: Optional[int],
    margin: Optional[float],
    alpha: Optional[float],
    method: Optional[str],
    lr: Optional[float],
    weight_decay: Optional[float],
    center_lr: Optional[float],
    seed: Optional[int],
    device: Optional[str]
) -> ArgumentParser:
    
    parser = ArgumentParser(description="UTMap Model Training")
    parser.add_argument('--model_name', default='model' if model_name is None else model_name, type=str, help='Name of the training model')
    parser.add_argument('--num_classes', default=2 if num_classes is None else num_classes, type=int, help='Number of the classes in dataset')
    parser.add_argument('--loss', default='tripletcenterloss' if loss is None else loss, type=str, help='Define loss type')
    parser.add_argument('--feature_dim', default=2 if feature_dim is None else feature_dim, type=int, help='Number of dimensions in IUS')
    parser.add_argument('--epochs', default=10 if epochs is None else epochs, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=16 if batch_size is None else batch_size, type=int, help='Batch size')
    parser.add_argument('--margin', default=5.0 if margin is None else margin, type=float, help='Margin for loss triplet loss function')
    parser.add_argument('--alpha', default=0.01 if alpha is None else alpha, type=float, help='Alpha parameter in T.C.Softmax Loss')
    parser.add_argument('--method', default='cmatrix' if method is None else method, type=str, help='Chosen method to create errros classes')
    parser.add_argument('--lr', default=1e-4 if lr is None else lr, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4 if weight_decay is None else weight_decay, type=float, help='Weight decay for model optim')
    parser.add_argument('--center_lr', default=1e-1 if center_lr is None else center_lr, type=float, help='Learning rate for centers param')
    parser.add_argument('--seed', default=np.random.randint(100_000) if seed is None else seed, type=int, help='Seed for reproducibility')
    parser.add_argument('--device', default='cpu' if device is None else device, type=str, help='Device for training (e.g., cpu or cuda)')

    return parser


def run_train_epoch(
    utmap: UTMap,
    backbone: BackboneResNet18,
    dataloader: DataLoader,
    label_mapping: dict,
    optimizer: Callable,
    criterion: Callable,
    optimizer_center: Callable,
    device: str,
    th: float = 0.5
) -> float:

    utmap.train()
    backbone.eval()

    total_loss = 0.0
    num_batchs = len(dataloader)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #   Get Feature Maps, Predicts and create Modified Labels
        with no_grad():
            features, logits = backbone.get_featuremaps_and_logits(inputs)
            predicts = where(logits.sigmoid() < th, 0, 1)

        modified_labels = create_modified_labels(label_mapping, labels, predicts).to(device)

        optimizer.zero_grad()
        optimizer_center.zero_grad()

        embeddings = utmap(features)

        loss = criterion(embeddings, modified_labels)
        loss.backward()

        if criterion.centers.grad is not None:
            clip_grad_value_(criterion.centers, clip_value=0.01)

        optimizer.step()
        optimizer_center.step()
        total_loss += loss.item()

    total_loss /= num_batchs
    return total_loss


def run_val_epoch(
    utmap: UTMap, 
    backbone: BackboneResNet18,
    dataloader: DataLoader, 
    label_mapping: dict,
    criterion: Callable, 
    device: str,
    th: float = 0.5
) -> float:
    
    utmap.eval()
    backbone.eval()

    total_loss = 0.0
    num_batchs = len(dataloader)

    with no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            features, logits = backbone.get_featuremaps_and_logits(inputs)
            predicts = where(logits.sigmoid() < th, 0, 1)
                
            modified_labels = create_modified_labels(label_mapping, labels, predicts).to(device)

            outputs = utmap(features)

            loss = criterion(outputs, modified_labels)
            total_loss += loss.item()

        total_loss /= num_batchs

    return total_loss


def do_utmap_training(
    args: Namespace, 
    utmap: UTMap,
    backbone: BackboneResNet18, 
    label_mapping: dict,
    num_modified_classes: int,
    train_dataset: Dataset, 
    val_dataset: Dataset
): 
    
    device = args.device
    progress_bar = trange(args.epochs, desc=f'Train {args.model_name}')

    backbone.eval()
    backbone.to(device)
    utmap.to(device)
    
    #########################
    # Create DataLoaders
    #########################
    train_generator = Generator().manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        generator=train_generator,
        shuffle=True
    )
    val_dataloader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    ####################################################
    # Set Optimizer, Loss, Checkpoint and Metric Logger
    ####################################################    
    if args.loss == "tripletcenterloss":
        criterion = TripletCenterLoss(
            num_classes=num_modified_classes, 
            feature_dim=args.feature_dim, 
            margin=args.margin, 
            device=device
        )
    
    elif args.loss == "tripletcentersoftmaxloss":
        criterion = TripletCenterSoftmaxLoss(
            num_classes=num_modified_classes, 
            feature_dim=args.feature_dim, 
            margin=args.margin, 
            alpha=args.alpha,
            device=device
        )

    optimizer = Adam(utmap.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_center = Adam(criterion.parameters(), lr=args.center_lr)

    logger = MetricsLogger(prefix=args.model_name)
    logger.add_metrics(['train_loss', 'val_loss'])

    checkpoint = Checkpoint(
        args=args, 
        metric_name='val_loss',
        initial_value=np.inf,
        output_dir='src'
    )

    for epoch in progress_bar:
        ##################
        # Run Train Epoch
        ##################
        train_loss = run_train_epoch(
            utmap=utmap,
            backbone=backbone,
            dataloader=train_dataloader,
            label_mapping=label_mapping,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            criterion=criterion,
            device=device
        )

        ################
        # Run Val Epoch
        ################
        val_loss = run_val_epoch(
            utmap=utmap,
            backbone=backbone,
            dataloader=val_dataloader,
            label_mapping=label_mapping,
            criterion=criterion,
            device=device
        )

        ###########################
        # Get Stats and Checkpoint
        ###########################
        stats = {
            'epoch': epoch,
            'train_loss':train_loss,
            'val_loss': val_loss,
        }

        if checkpoint.value > val_loss:
            checkpoint(model=utmap, stats=stats)   
        logger.update(stats)

        progress_bar.set_postfix({
                'Train':train_loss,
                'Val':val_loss
        })
    
    backbone = backbone.to('cpu')
    utmap = utmap.to('cpu')
    checkpoint.save_dict.update({'centers':criterion.centers.clone().detach().to('cpu')})

    return logger.log_dict, checkpoint.save_dict
