import numpy as np

from argparse import Namespace, ArgumentParser

from torch import no_grad, where, Generator
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from tqdm import trange
from typing import Callable, Optional

from ..utils.model_checkpoint import Checkpoint
from ..utils.metrics_logger import MetricsLogger


def create_classifier_argparse(
    model_name: Optional[str],
    num_classes: Optional[int],
    epochs: Optional[int],
    batch_size: Optional[int],
    lr: Optional[float],
    seed: Optional[int],
    device: Optional[str],
) -> ArgumentParser:
    
    parser = ArgumentParser(description="Classifier Model Training")
    parser.add_argument('--model_name', default='model' if model_name is None else model_name, type=str, help='Name of the training model')
    parser.add_argument('--num_classes', default=2 if num_classes is None else num_classes, type=int, help='Number of the classes in dataset')
    parser.add_argument('--epochs', default=100 if epochs is None else epochs, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64 if batch_size is None else batch_size, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-5 if lr is None else lr, type=float, help='Learning rate')
    parser.add_argument('--seed', default=np.random.randint(100) if seed is None else seed, type=int, help='Seed for reproducibility')
    parser.add_argument('--device', default='cpu' if device is None else device, type=str, help='Device for training (e.g., cpu or cuda)')

    return parser


def run_train_epoch(
    model: Callable,
    dataloader: DataLoader,
    optimizer: Callable,
    criterion: Callable,
    device: str,
    th: float = 0.5
) -> float:

    model.train()

    total_loss = 0.
    accuracy = 0.
    num_samples = 0
    
    num_batchs = len(dataloader)

    for inputs, labels in dataloader:
        
        labels = labels.float()
        labels = labels.unsqueeze(1) if labels.dim() == 1 else labels

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        outputs = outputs.detach()
        predicts = where(outputs.sigmoid() < th, 0, 1)

        accuracy += (predicts == labels).sum().item()
        num_samples += len(predicts)

    total_loss /= num_batchs
    accuracy /= num_samples

    return total_loss, accuracy


def run_val_epoch(
    model: Callable,
    dataloader: DataLoader,
    criterion: Callable,
    device: str,
    th: float = 0.5
) -> float:

    model.eval()

    total_loss = 0.
    accuracy = 0.
    num_samples = 0
    
    num_batchs = len(dataloader)

    with no_grad():
        for inputs, labels in dataloader:
            
            labels = labels.float()
            labels = labels.unsqueeze(1) if labels.dim() == 1 else labels

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            
            outputs = outputs.detach()
            predicts = where(outputs.sigmoid() < th, 0, 1)

            accuracy += (predicts == labels).sum().item()
            num_samples += len(predicts)

    total_loss /= num_batchs
    accuracy /= num_samples

    return total_loss, accuracy


def do_classifier_training(
        args: Namespace, 
        model: Callable, 
        train_dataset: Dataset,
        val_dataset: Dataset,
): 

    device = args.device

    model.train()
    model.to(device)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    t = trange(args.epochs, desc=f'Train {args.model_name}')
    t.set_postfix({'Train':np.inf, 'Val':np.inf})
    
    logger = MetricsLogger(prefix=args.model_name)
    logger.add_metrics(['train_loss', 'train_accuracy'])
    logger.add_metrics(['val_loss', 'val_accuracy'])

    checkpoint = Checkpoint(
        args=args, 
        metric_name='val_loss',
        initial_value=np.inf,
        output_dir='src'
    )

    train_generator = Generator().manual_seed(args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=train_generator)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    for epoch in t:
        
        #############
        # Train Data
        #############
        train_loss, train_acc = run_train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        ###########
        # Val Data
        ###########
        val_loss, val_acc = run_val_epoch(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device
        )

        ###########################
        # Get Stats and Checkpoint
        ###########################
        stats = {
            'epoch': epoch,
            'train_loss':train_loss,
            'train_accuracy':train_acc,
            'val_loss': val_loss,
            'val_accuracy':val_acc
        }

        if checkpoint.value > val_loss:
            checkpoint(model=model, stats=stats)   
        logger.update(stats)

        t.set_postfix(
            {
                'Train':train_loss,
                'Val':val_loss
            }
        )

    model.to('cpu')
    return logger.log_dict, checkpoint.save_dict
