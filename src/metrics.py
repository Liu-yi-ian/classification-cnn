"""Compute validation and test metrics from cross-entropy and a confusion matrix."""
import torch
from torch.nn import functional as F

from .data import NAMES


class Metrics:
    """Accumulate counts and loss sums without averaging batch means."""
    def __init__(self, classes, device, twelve_class_metrics=True):
        self.classes = classes
        self.twelve_class_metrics = twelve_class_metrics
        self.cm = torch.zeros(classes, classes, dtype=torch.long, device=device)
        self.ce = torch.zeros((), dtype=torch.float64, device=device)
        self.objective = torch.zeros_like(self.ce)

    def update(self, logits, labels, ce, objective=None):
        """Store detached predictions so metrics do not retain training graphs."""
        prediction = logits.detach().argmax(1)
        self.cm += torch.bincount(labels * self.classes + prediction,
                                  minlength=self.classes ** 2).reshape(self.classes, self.classes)
        self.ce += ce.detach().double().sum()
        self.objective += (ce if objective is None else objective).detach().double().sum()

    def result(self):
        """Report raw accuracy and, for twelve classes, class-balanced recall."""
        cm = self.cm.cpu()
        support = cm.sum(1)
        count = int(cm.sum())
        result = dict(count=count, ce=float(self.ce) / count,
                      objective_ce=float(self.objective) / count,
                      accuracy=float(cm.diag().sum()) / count)
        if self.twelve_class_metrics:
            recall = cm.diag().double() / support.clamp_min(1)
            # Insect metrics use the same twelve-class prediction as all other
            # images; the last two confusion-matrix rows select ants and bees.
            result.update(macro_recall=float(recall.mean()),
                          insect_accuracy=float(cm.diag()[10:].sum()) / max(1, int(support[10:].sum())),
                          per_class={name: dict(recall=float(r), support=int(n))
                                     for name, r, n in zip(NAMES, recall, support)},
                          confusion_matrix=cm.tolist())
        return result


@torch.inference_mode()
def evaluate(model, loader, twelve_class_metrics=True):
    """Evaluate every validation or test sample once without augmentation or gradients."""
    model.eval()
    device = next(model.parameters()).device
    meter = Metrics(model.num_classes, device, twelve_class_metrics)
    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(images)
        ce = F.cross_entropy(logits.float(), labels, reduction='none')
        meter.update(logits, labels, ce)
    return meter.result()
