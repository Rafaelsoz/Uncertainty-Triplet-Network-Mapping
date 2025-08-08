from torch import tensor


def create_label_mapping(num_classes: int) -> dict:
    label_mapping = {}
    new_classes = set()

    for i in range(num_classes):
        new_classes.add(i)

        for j in range(num_classes):
            if i == j:
              label_mapping.update({(i, j):i})

            else:
              label_mapping.update({(i, j): num_classes})
    
    new_classes.add(num_classes)
    return label_mapping, list(new_classes)


def create_modified_labels(
    label_mapping: dict,
    labels: tensor,
    predicts: tensor
):
    labels, predicts = labels.detach(), predicts.detach()

    labels = labels.squeeze() if labels.dim() > 1 else labels
    predicts = predicts.squeeze() if predicts.dim() > 1 else predicts

    modified_labels = tensor([label_mapping[(y.item(), y_hat.item())] for y, y_hat in zip(labels, predicts)])

    return modified_labels