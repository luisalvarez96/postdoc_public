from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from matplotlib import pyplot
import os
import numpy as np
from sklearn.preprocessing import label_binarize

def AUROC(scores, labels, save_path=None):
    """
    Calculates and plots AUROC for multi-class classification.

    Args:
        scores (torch.Tensor or np.ndarray): Predicted scores (probabilities).
                                             Shape: (n_samples, n_classes)
        labels (torch.Tensor or np.ndarray): True labels.
                                             Shape: (n_samples,)
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    # Ensure inputs are numpy arrays
    if not isinstance(scores, np.ndarray):
        scores = scores.cpu().numpy()
    if not isinstance(labels, np.ndarray):
        labels = labels.cpu().numpy()

    metrics = []

    # Check if this is a binary classification scenario
    if scores.shape[1] == 1:
        # Calculate AUC and plot for binary case
        lr_auc = roc_auc_score(labels, scores)
        metrics.append({'Metric': 'AUROC', 'Value': lr_auc})
        write_to_out(f'AUROC: {lr_auc:.4f}', save_path) if save_path is not None else None

        if save_path is not None:
            ns_probs = [0 for _ in range(len(labels))]
            ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)
            lr_fpr, lr_tpr, _ = roc_curve(labels, scores)
            
            pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            pyplot.plot(lr_fpr, lr_tpr, label='Model')
    else:
        # Multi-class case
        n_classes = scores.shape[1]

        # calculate AUROC for each class.
        # converting labels to one-hot encoding for roc_auc_score.
        labels_one_hot = label_binarize(labels, classes=range(n_classes))

        # Calculate AUROC with both 'micro' and 'macro' averages
        lr_auc_micro = roc_auc_score(labels_one_hot, scores, multi_class='ovr', average='micro')
        lr_auc_macro = roc_auc_score(labels_one_hot, scores, multi_class='ovr', average='macro')

        metrics.append({'Metric': 'Micro-AUROC', 'Value': lr_auc_micro})
        metrics.append({'Metric': 'Macro-AUROC', 'Value': lr_auc_macro})

        if save_path is not None:
            write_to_out(f'Micro-AUROC: {lr_auc_micro:.4f}', save_path)
            write_to_out(f'Macro-AUROC: {lr_auc_macro:.4f}', save_path)

            # plot the micro-average curve and curves for each class (OvR).

            # Calculate FPR and TPR for each class (One vs Rest)
            fpr = dict()
            tpr = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(labels_one_hot[:, i], scores[:, i])
                pyplot.plot(fpr[i], tpr[i], linestyle=':', label=f'Class {i} AUROC: {roc_auc_score(labels_one_hot[:, i], scores[:, i]):.3f}')
            
            # Calculate the micro-average ROC curve
            fpr_micro, tpr_micro, _ = roc_curve(labels_one_hot.ravel(), scores.ravel())
            pyplot.plot(fpr_micro, tpr_micro, linestyle='-', label=f'Micro-average AUROC: {lr_auc_micro:.3f}', color='deeppink', lw=2)
            pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill', color='gray')


    if save_path is not None:
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('Receiver Operating Characteristic (ROC) Curve')
        pyplot.legend()
        
        final_save_path = save_path + '/AUROC.png' if save_path is not None else '../out/AUROC.png'
        pyplot.savefig(final_save_path, dpi=180)
        pyplot.clf()

    return metrics

def AUPR(scores, labels, save_path=None):
    """
    Calculates and plots AUPR for multi-class classification.

    Args:
        scores (torch.Tensor or np.ndarray): Predicted scores (probabilities).
                                             Shape: (n_samples, n_classes)
        labels (torch.Tensor or np.ndarray): True labels.
                                             Shape: (n_samples,)
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if not isinstance(scores, np.ndarray):
        scores = scores.cpu().numpy()
    if not isinstance(labels, np.ndarray):
        labels = labels.cpu().numpy()

    metrics = []

    if scores.shape[1] == 1:
        # Binary case
        lr_precision, lr_recall, _ = precision_recall_curve(labels, scores)
        lr_auc = auc(lr_recall, lr_precision)
        metrics.append({'Metric': 'AUPR', 'Value': lr_auc})
        write_to_out(f'AUPR: {lr_auc:.4f}', save_path) if save_path is not None else None

        if save_path is not None:
            no_skill = len(labels[labels == 1]) / len(labels)
            pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
            pyplot.plot(lr_recall, lr_precision, label='Model')
    else:
        # Multi-class case
        n_classes = scores.shape[1]
        labels_one_hot = label_binarize(labels, classes=range(n_classes))
        
        # Calculate micro-average AUPR (a global metric)
        precision_micro, recall_micro, _ = precision_recall_curve(labels_one_hot.ravel(), scores.ravel())
        aupr_micro = auc(recall_micro, precision_micro)
        metrics.append({'Metric': 'Micro-AUPR', 'Value': aupr_micro})
        write_to_out(f'Micro-AUPR: {aupr_micro:.4f}', save_path) if save_path is not None else None

        # Calculate AUPR for each class and store them for a macro average
        auprs = []
        for i in range(n_classes):
            precision_i, recall_i, _ = precision_recall_curve(labels_one_hot[:, i], scores[:, i])
            aupr_i = auc(recall_i, precision_i)
            auprs.append(aupr_i)
            if save_path is not None:
                pyplot.plot(recall_i, precision_i, linestyle=':', label=f'Class {i} AUPR: {aupr_i:.3f}')

        aupr_macro = np.mean(auprs)
        metrics.append({'Metric': 'Macro-AUPR', 'Value': aupr_macro})
        write_to_out(f'Macro-AUPR: {aupr_macro:.4f}', save_path) if save_path is not None else None

        if save_path is not None:
            # Plot the micro-average curve
            pyplot.plot(recall_micro, precision_micro, linestyle='-', label=f'Micro-average AUPR: {aupr_micro:.3f}', color='deeppink', lw=2)

            # Plot the "No Skill" line for each class (based on class imbalance)
            for i in range(n_classes):
                no_skill_i = np.sum(labels == i) / len(labels)
                pyplot.plot([0, 1], [no_skill_i, no_skill_i], linestyle='--', color='gray', alpha=0.5)
            # A single no-skill line is not ideal for multi-class due to varying class balance.

    if save_path is not None:
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.title('Precision-Recall Curve')
        pyplot.legend()
        
        final_save_path = save_path + '/AUPR.png'
        pyplot.savefig(final_save_path, dpi=180)
        pyplot.clf()

    return metrics

def plot_losses(losses, val_losses, save_path = None):
    pyplot.plot(range(len(losses)), losses, label="loss")
    if len(val_losses) > 0:
        pyplot.plot(range(len(losses)), val_losses, label="val_loss")
        pyplot.legend()
    if save_path is not None:
        save_path = save_path + '/losses'
    else:
        save_path = '../out/losses'
    pyplot.savefig(save_path, dpi=200)
    pyplot.clf()

def write_to_out(text, out_path = '../out'):
    with open(out_path + '/out.txt', 'a') as f:
        f.write(text + '\n')
