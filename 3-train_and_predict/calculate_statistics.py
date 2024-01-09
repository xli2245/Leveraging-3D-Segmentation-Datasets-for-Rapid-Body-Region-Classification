import torch

# Assuming 'preds' and 'labels' are predictions and true labels respectively
def calculate_statistics(pred, label, tag='val'):
    pred = torch.sigmoid(pred)

    # exclude the background class
    preds = pred[0, 1:, :, :]
    labels = label[0, 1:, :, :]

    # Binarize predictions
    preds_bin = (preds > 0.5).float()

    # Compute true/false positives/negatives
    tp = (preds_bin * labels).sum(dim=[1,2])
    fp = (preds_bin * (1 - labels)).sum(dim=[1,2])
    fn = ((1 - preds_bin) * labels).sum(dim=[1,2])
    tn = ((1 - preds_bin) * (1 - labels)).sum(dim=[1,2])

    # Compute accuracy, precision, recall, F1 for each class
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Calculate micro-averaged precision, recall, F1
    micro_accuracy = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum())
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10)

    # Calculate macro-averaged precision, recall, F1
    macro_accuracy = accuracy.mean()
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    micro = [micro_accuracy.item(), micro_precision.item(), micro_recall.item(), micro_f1.item()]
    macro = [macro_accuracy.item(), macro_precision.item(), macro_recall.item(), macro_f1.item()]
    if tag == 'val':
        return micro, macro
    elif tag == 'test':
        return tp, fp, fn, tn, micro, macro
    else:
        raise ValueError(f'invalid tag = {tag}')

