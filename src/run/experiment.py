import numpy as np

from copy import deepcopy
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from torch import no_grad, where, cat, max, stack, log, tensor
from torchvision.models import ResNet
from torch.nn import Module, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from ..utils.seed import define_seed
from ..classifiers import load_resnet18_dropout, get_resnet_dropout_predicts, ResNet18Dropout
from ..utmaps import load_utmap, UTMap

from ..datasets.dataset import ImageDataset

from ..classifiers import get_resnet18, BackboneResNet18, get_resnet_dropout_predicts
from ..utils.label_mapping import create_label_mapping
from ..train.classifier_training import do_classifier_training, create_classifier_argparse
from ..train.utmap_training import do_utmap_training, create_utmap_argparse
from ..metrics import FailureMetrics
from ..scores import TrustScore, NeighborhoodReliabilityScore


def get_classifier_parser(configs: dict, use_mcdo: bool = False):
    classifier_parser = create_classifier_argparse(
        model_name=configs['resnet18']['model_name'],
        num_classes=configs['data']['num_classes'],
        epochs= configs['resnet18']['mcdo_epochs'] if use_mcdo else configs['resnet18']['epochs'],
        batch_size= configs['resnet18']['mcdo_batch_size'] if use_mcdo else configs['resnet18']['batch_size'],
        lr= configs['resnet18']['mcdo_lr'] if use_mcdo else configs['resnet18']['lr'],
        seed=configs['env']['seed'],
        device=configs['env']['device'],
    )

    return classifier_parser


def get_utmap_parser(configs: dict):
    utmap_parser = create_utmap_argparse(
        model_name=configs['utmap']['model_name'],
        num_classes=configs['data']['num_classes'],
        loss=configs['utmap']['loss'],
        feature_dim=configs['utmap']['feature_dim'],
        epochs=configs['utmap']['epochs'],
        batch_size=configs['utmap']['batch_size'],
        margin=configs['utmap']['margin'],
        alpha=configs['utmap']['alpha'],
        method=configs['utmap']['method'],
        lr=configs['utmap']['lr'],
        weight_decay=configs['utmap']['weight_decay'],
        center_lr=configs['utmap']['center_lr'],
        seed=configs['env']['seed'],
        device=configs['env']['device'],
    )

    return utmap_parser


def get_embeddings_and_stats(
    utmap: UTMap,
    backbone: BackboneResNet18,
    dataset: Dataset,
    batch_size: int = 128,
    device: str = 'cpu',
    th: float = 0.5
):
    utmap.eval()
    utmap.to(device)
    
    backbone.eval()
    backbone.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    backbone_featuremaps = []
    backbone_embeddings = []
    utmap_projection = []
    entropys = []
    probabilities = []
    predictions = []
    targets = []

    progress_bar = tqdm(dataloader)

    criterion = BCEWithLogitsLoss(reduction='none')
    
    with no_grad():
        for inputs, labels in progress_bar:

            labels = labels.float()
            labels = labels.unsqueeze(1) if labels.dim() == 1 else labels
        
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze()

            featuremaps, embeddings, logits = backbone.get_featuremaps_embeddings_and_logits(inputs)
            projection = utmap(featuremaps)

            if isinstance(projection, tuple):
                projection = projection[0]

            logits = logits.squeeze()
            probs = logits.sigmoid()
            predicts = where(probs < th, 0, 1)
            entropy = criterion(logits, labels)
            
            backbone_featuremaps.append(featuremaps.cpu())
            backbone_embeddings.append(embeddings.cpu())
            utmap_projection.append(projection.cpu())
            entropys.append(entropy.cpu())
            probabilities.append(probs.cpu())
            predictions.append(predicts.cpu())
            targets.append(labels.cpu())

    backbone_featuremaps = cat(backbone_featuremaps)
    backbone_embeddings = cat(backbone_embeddings)
    utmap_projection = cat(utmap_projection)
    entropys = cat(entropys)
    probabilities = cat(probabilities)
    predictions = cat(predictions)
    targets = cat(targets)

    max_class_probability = max(probabilities, 1 - probabilities)

    return backbone_featuremaps, backbone_embeddings, utmap_projection, entropys, probabilities, predictions, targets, max_class_probability


def get_scores_and_metrics(
    utmap: UTMap,
    classifier: ResNet,
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 128,
    device: str = 'cpu'
):
    
    #############################
    #   Get Emdeddings and Stats
    #############################
    backbone = BackboneResNet18(
        model_state=classifier.state_dict(), 
        requires_grad=False
    )

    _, train_embeddings, train_projection, _, _, _, train_targets, _ = get_embeddings_and_stats(
        utmap=utmap,
        backbone=backbone,
        dataset=train_dataset,
        batch_size=batch_size,
        device=device
    )

    _, test_embeddings, test_projection, _, _, test_predictions, test_targets, test_mcp = get_embeddings_and_stats(
        utmap=utmap,
        backbone=backbone,
        dataset=test_dataset,
        batch_size=batch_size,
        device=device
    )

    ###############
    #   Get Scores
    ###############
    ts = TrustScore()
    ts.fit(train_embeddings.numpy(), train_targets.numpy().astype(int))
    ts_values = ts.get_score(test_embeddings.numpy(), test_predictions.numpy().astype(int))

    nrs = NeighborhoodReliabilityScore()
    nrs.fit(train_projection.numpy(), train_targets.numpy())
    nrs_values = nrs.get_score(test_projection.numpy(), test_predictions.numpy().astype(int))
    
    ################
    #   Get Metrics
    ################
    failure_metrics = FailureMetrics()
    mcp_metrics = failure_metrics.get_scores(test_predictions.numpy(), test_targets.numpy(), test_mcp.numpy())
    ts_metrics = failure_metrics.get_scores(test_predictions.numpy(), test_targets.numpy(), ts_values)
    nrs_metrics = failure_metrics.get_scores(test_predictions.numpy(), test_targets.numpy(), nrs_values)

    scores_results = {
        "MCP":{
            "values":test_mcp,
            "metrics":mcp_metrics
        },
        "TS":{
            "values":ts_values,
            "metrics":ts_metrics
        },
        "NRS":{
            "values":nrs_values,
            "metrics":nrs_metrics
        }
    }

    return scores_results


def get_mcdo_scores_and_metrics(
    num_classes: int,
    classifier: ResNet18Dropout,
    dataset: Dataset,
    num_samples: int,
    device: str = 'cpu'  
):
    mcdo_targets, mcdo_predicts, mcdo_entropy  = get_resnet_dropout_predicts(
        num_classes=num_classes,
        classifier=classifier,
        dataset=dataset,
        num_samples=num_samples,
        device=device
    )

    failure_metrics = FailureMetrics()
    mcdo_metrics = failure_metrics.get_scores(mcdo_predicts.numpy(), mcdo_targets.numpy(), mcdo_entropy.numpy())

    return {
        "values":mcdo_entropy,
        "metrics":mcdo_metrics
    }
    

def run_experiment(
    configs: dict, 
    train_dataset: Dataset, 
    val_dataset: Dataset,
    aug_train_dataset: Optional[Callable] = None,
    classifier: Optional[Module] = None,
):
    #############
    #   Set Seed
    #############
    define_seed(configs['env']['seed'])

    ################
    #   Create args 
    ################
    classifier_parser = get_classifier_parser(configs=configs)
    classifier_args, _ = classifier_parser.parse_known_args()

    utmap_parser = get_utmap_parser(configs=configs)
    utmap_args, _ = utmap_parser.parse_known_args()
    

    if classifier is None:
        #####################
        #   Train Classifier 
        #####################
        classifier = get_resnet18(output_size=1, freeze_layers=False)
        clf_logger_dict, clf_best_stats = do_classifier_training(
            args=classifier_args,
            model=classifier,
            train_dataset=train_dataset if aug_train_dataset is None else aug_train_dataset,
            val_dataset=val_dataset
        )

        classifier_state = clf_best_stats['model_state']

    else:
        clf_logger_dict, clf_best_stats = {}, {}
        classifier_state = classifier.state_dict()


    #######################
    #   Create new classes
    #######################
    backbone = BackboneResNet18(
        model_state=classifier_state, 
        requires_grad=False
    )
    label_mapping, modified_classes = create_label_mapping(configs['data']['num_classes'])


    #################
    #   Train UTMap 
    #################
    utmap = UTMap(
        feature_dim=configs['utmap']['feature_dim'],
        output_size=None if configs['utmap']['loss'] == 'tripletcenterloss' else len(modified_classes)
    )
     
    utmap_logger_dict, utmap_best_stats = do_utmap_training(
        args=utmap_args,
        utmap=utmap,
        backbone=backbone,
        label_mapping=label_mapping,
        num_modified_classes=len(modified_classes),
        train_dataset=train_dataset if aug_train_dataset is None else aug_train_dataset,
        val_dataset=val_dataset
    )

    classifier_infos = {
        "logger_dict":clf_logger_dict,
        "best_stats":clf_best_stats
    }

    utmap_infos = {
        "label_mapping":label_mapping, 
        "modified_classes":modified_classes,
        "logger_dict": utmap_logger_dict,
        "best_stats": utmap_best_stats
    }

    return classifier_infos, utmap_infos 

def get_ensemble_score(
    models: list, 
    dataset: Dataset,
    device: str = "cpu",
    th: float = 0.5
):
    models = [model.to(device) for model in models]
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    h_max = log(tensor(2.0)).to(device) 
    
    entropies = []
    predicts = []
    targets = []
    
    with no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)

            all_probs = []
            for model in models:
                model.eval()
                logits = model(inputs).squeeze()  
                probs = logits.sigmoid()  
                all_probs.append(probs)
            
            all_probs = stack(all_probs, dim=0)
            mean_probs = all_probs.mean(dim=0)  

            probs = stack([1 - mean_probs, mean_probs], dim=1)

            entropy = (probs * log(probs + 1e-9)).sum(dim=1) / h_max
            entropy = entropy.squeeze() if entropy.dim() > 1 else entropy
            entropies.append(entropy.cpu())
            
            predict = where(mean_probs < th, 0, 1).cpu()
            predict = predict.squeeze() if predict.dim() > 1 else predict
            predicts.append(predict)

            labels = labels.squeeze() if labels.dim() > 1 else labels
            targets.append(labels.cpu()) 
    
    predicts = cat(predicts)
    targets = cat(targets)
    entropies = cat(entropies)

    failure_metrics = FailureMetrics()
    de_metrics = failure_metrics.get_scores(predicts.numpy(), targets.numpy(), entropies.numpy())
    return {
        "values":entropies,
        "metrics":de_metrics
    }


def run_kfold_experiment(
    configs: dict, 
    paths: np.ndarray, 
    targets: np.ndarray, 
    class_to_idx: dict, 
    transform: Callable, 
    aug_transform: Callable,
    n_splits: int = 10, 
    show_metrics: bool = True,
    metrics_list: list = ['roc_auc_score', 'aupr_error', 'fpr_at_0.95tpr', 'aurc', 'eaurc'],
    M: int = 5
):
    #############
    #   Set Seed
    #############
    define_seed(configs['env']['seed'])

    ################
    #   Create args 
    ################
    classifier_parser = get_classifier_parser(configs=configs, use_mcdo=False)
    classifier_args, _ = classifier_parser.parse_known_args()

    mcdo_classifier_parser = get_classifier_parser(configs=configs, use_mcdo=True)
    mcdo_classifier_args, _ = mcdo_classifier_parser.parse_known_args()

    utmap_parser = get_utmap_parser(configs=configs)
    utmap_args, _ = utmap_parser.parse_known_args()
    

    ######################
    #   Stratified K Fold
    ######################
    skf = StratifiedKFold(n_splits=n_splits, random_state=configs['env']['seed'], shuffle=True)

    results = {}
    for k_idx, (train_index, test_index) in enumerate(skf.split(paths, targets)):
        ################
        #   Split Data
        ################
        test_paths, test_targets = paths[test_index], targets[test_index]
        aux_train_paths, aux_train_targets = paths[train_index], targets[train_index]
        train_paths, val_paths, train_targets, val_targets = train_test_split(
            aux_train_paths, 
            aux_train_targets, 
            test_size=configs['data']['val_size'], 
            random_state=configs['env']['seed'], 
            stratify=aux_train_targets
        )

        ####################
        #   Create Datasets
        ####################
        aug_train_dataset = ImageDataset(train_paths, train_targets, class_to_idx, aug_transform)
        train_dataset = ImageDataset(train_paths, train_targets, class_to_idx, transform)
        val_dataset = ImageDataset(val_paths, val_targets, class_to_idx, transform)
        test_dataset = ImageDataset(test_paths, test_targets, class_to_idx, transform)

        #########################
        #   Train Deep Ensemble
        #########################
        ensemble_models = []

        # for i, current_dataset in enumerate(bootstrapped_datasets):
        for i in range(M):
            current_configs = deepcopy(configs)
            current_configs['env']['seed'] += i + 1

            model_parser = get_classifier_parser(configs=current_configs, use_mcdo=False)
            model_args, _ = model_parser.parse_known_args()

            model = get_resnet18(output_size=1, freeze_layers=False)
            clf_logger_dict, clf_best_stats = do_classifier_training(
                args=model_args,
                model=model,
                train_dataset=aug_train_dataset,
                val_dataset=val_dataset
            )
            model.load_state_dict(clf_best_stats['model_state'])
            ensemble_models.append(model)
        de_scores = get_ensemble_score(ensemble_models, test_dataset, configs['env']['device'])
            
        ##########################################
        #   Set Classifier and Dropout Classifier 
        ##########################################
        classifier = get_resnet18(output_size=1, freeze_layers=False)
        mcdo_classifier = load_resnet18_dropout(classifier.state_dict(), freeze_layers=False)

        #############################
        #   Train Dropout Classifier
        #############################
        mcdo_clf_logger_dict, mcdo_clf_best_stats = do_classifier_training(
            args=mcdo_classifier_args,
            model=mcdo_classifier,
            train_dataset=aug_train_dataset,
            val_dataset=val_dataset
        )
        mcdo_classifier.load_state_dict(mcdo_clf_best_stats['model_state'])
        mcdo_scores = get_mcdo_scores_and_metrics(
            num_classes=configs['data']['num_classes'],
            classifier=mcdo_classifier,
            dataset=test_dataset,
            num_samples=50,
            device=configs['env']['device']
        )

        ######################
        #   Train Classifier
        ######################
        clf_logger_dict, clf_best_stats = do_classifier_training(
            args=classifier_args,
            model=classifier,
            train_dataset=aug_train_dataset,
            val_dataset=val_dataset
        )
        classifier.load_state_dict(clf_best_stats['model_state'])
        backbone = BackboneResNet18(
            model_state=clf_best_stats['model_state'], 
            requires_grad=False
        )
        
        #######################
        #   Create new classes
        #######################
        label_mapping, modified_classes = create_label_mapping(configs['data']['num_classes'])
        
        #################
        #   Train UTMap 
        #################
        utmap = UTMap(
            feature_dim=configs['utmap']['feature_dim'],
            output_size=None if configs['utmap']['loss'] == 'tripletcenterloss' else len(modified_classes)
        )
        
        utmap_logger_dict, utmap_best_stats = do_utmap_training(
            args=utmap_args,
            utmap=utmap,
            backbone=backbone,
            label_mapping=label_mapping,
            num_modified_classes=len(modified_classes),
            train_dataset=aug_train_dataset,
            val_dataset=val_dataset
        )
        utmap = load_utmap(utmap_best_stats['model_state']) 

        ###########################
        #   Get Scores and metrics 
        ###########################
        scores = get_scores_and_metrics(
            utmap=utmap,
            classifier=classifier,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=configs['env']['device']
        )
        scores["MCDO"] = mcdo_scores
        scores["DE"] = de_scores

        if show_metrics:
            for score in scores:
                current_score = scores[score]['metrics']
                print('=' * 55)
                print(f'\t\t\t{score}')
                for current_metric in current_score:
                    if current_metric in metrics_list:
                        print(f'{current_metric.ljust(20).replace("_", " ").capitalize()} :: \t {current_score[current_metric]}')
                print('=' * 55)

        results[f"split_{k_idx + 1}"] = {
            "scores":scores,
            "clf_logger":clf_logger_dict,
            "dpt_clf_logger":mcdo_clf_logger_dict,
            "utmap_logger":utmap_logger_dict
        }
    
    return results


def run_hyperparameters_experiment(
    alphas: list,
    margins: list,
    configs: dict, 
    paths: np.ndarray, 
    targets: np.ndarray, 
    class_to_idx: dict, 
    transform: Callable, 
    aug_transform: Callable,
    n_splits: int = 5, 
    show_metrics: bool = True,
    metrics_list: list = ['roc_auc_score', 'aupr_error', 'fpr_at_0.95tpr', 'aurc', 'eaurc'],
):
    #############
    #   Set Seed
    #############
    define_seed(configs['env']['seed'])


    data = {}
    for alpha in alphas:
        for margin in margins: 
            current_configs = deepcopy(configs)
            current_configs['utmap']['alpha'] = alpha
            current_configs['utmap']['margin'] = margin

            ################
            #   Create args 
            ################
            classifier_parser = get_classifier_parser(configs=current_configs, use_mcdo=False)
            classifier_args, _ = classifier_parser.parse_known_args()

            utmap_parser = get_utmap_parser(configs=current_configs)
            utmap_args, _ = utmap_parser.parse_known_args()
            
            ######################
            #   Stratified K Fold
            ######################
            skf = StratifiedKFold(n_splits=n_splits, random_state=current_configs['env']['seed'], shuffle=True)
            results = {}
                    
            for k_idx, (train_index, test_index) in enumerate(skf.split(paths, targets)):
                ################
                #   Split Data
                ################
                test_paths, test_targets = paths[test_index], targets[test_index]
                aux_train_paths, aux_train_targets = paths[train_index], targets[train_index]
                train_paths, val_paths, train_targets, val_targets = train_test_split(
                    aux_train_paths, 
                    aux_train_targets, 
                    test_size=current_configs['data']['val_size'], 
                    random_state=current_configs['env']['seed'], 
                    stratify=aux_train_targets
                )

                ####################
                #   Create Datasets
                ####################
                aug_train_dataset = ImageDataset(train_paths, train_targets, class_to_idx, aug_transform)
                train_dataset = ImageDataset(train_paths, train_targets, class_to_idx, transform)
                val_dataset = ImageDataset(val_paths, val_targets, class_to_idx, transform)
                test_dataset = ImageDataset(test_paths, test_targets, class_to_idx, transform)

                ####################
                #   Set Classifier  
                ####################
                classifier = get_resnet18(output_size=1, freeze_layers=False)
                
                ######################
                #   Train Classifier
                ######################
                _, clf_best_stats = do_classifier_training(
                    args=classifier_args,
                    model=classifier,
                    train_dataset=aug_train_dataset,
                    val_dataset=val_dataset
                )
                classifier.load_state_dict(clf_best_stats['model_state'])
                backbone = BackboneResNet18(
                    model_state=clf_best_stats['model_state'], 
                    requires_grad=False
                )
                
                #######################
                #   Create new classes
                #######################
                label_mapping, modified_classes = create_label_mapping(current_configs['data']['num_classes'])
                
                #################
                #   Train UTMap 
                #################
                utmap = UTMap(
                    feature_dim=current_configs['utmap']['feature_dim'],
                    output_size=None if current_configs['utmap']['loss'] == 'tripletcenterloss' else len(modified_classes)
                )
                
                _, utmap_best_stats = do_utmap_training(
                    args=utmap_args,
                    utmap=utmap,
                    backbone=backbone,
                    label_mapping=label_mapping,
                    num_modified_classes=len(modified_classes),
                    train_dataset=aug_train_dataset,
                    val_dataset=val_dataset
                )
                utmap = load_utmap(utmap_best_stats['model_state']) 

                ###########################
                #   Get Scores and metrics 
                ###########################
                scores = get_scores_and_metrics(
                    utmap=utmap,
                    classifier=classifier,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    device=current_configs['env']['device']
                )
                results[f"split_{k_idx + 1}"] = scores
            data[f'{alpha}_{margin}'] = results
    return data