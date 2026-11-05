"""Pretrain the full CNN, transfer its features, and fine-tune the twelve-class model."""
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ROOT, NAMES, ClassificationData, training_datasets
from .metrics import Metrics, evaluate
from .model import MultiResolutionCNN
from .sampling import BalancedSampler
from .state import setup, worker_seed, save_json, save_rng, restore_rng


def make_loader(dataset, config, training):
    """Shuffle ImageNet images or draw balanced downstream examples; keep evaluation ordered."""
    sampler = (BalancedSampler(dataset, config)
               if training and config['stage'] == 'finetune' else None)
    return DataLoader(dataset, batch_size=config['batch_size'],
                      shuffle=training and sampler is None, sampler=sampler,
                      num_workers=config['workers'], pin_memory=True,
                      worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(
                          config['seed'] + (1000 if training else 2000)))


def set_phase(model, phase):
    """Freeze the feature extractor for head adaptation, then train the full model."""
    model.train().requires_grad_(phase != 'head')
    if phase == 'head':
        # eval() keeps pretrained BatchNorm statistics fixed during head adaptation.
        # Only the classifier weights and its input dropout participate in training.
        model.eval()
        model.classifier.train().requires_grad_(True)
        model.dropout.train()


def optimizer_for(model, config, phase):
    """Use SGD for pretraining and AdamW with stage-specific parameter groups for transfer."""
    if phase == 'pretrain':
        return torch.optim.SGD([dict(params=list(model.parameters()), name='all', lr=config['lr'])],
                               momentum=config['momentum'], weight_decay=config['weight_decay'])
    if phase == 'head':
        groups = [dict(params=list(model.classifier.parameters()), name='new', lr=config['head_lr'])]
    else:
        groups = [dict(params=[p for name, p in model.named_parameters()
                              if not name.startswith('classifier.')],
                       name='pretrained', lr=config['backbone_lr']),
                  dict(params=list(model.classifier.parameters()), name='new', lr=config['new_lr'])]
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])


def train_epoch(model, loader, optimizer, config, phase):
    """Accumulate gradients over image batches and perform one update per effective batch."""
    set_phase(model, phase)
    device = next(model.parameters()).device
    meter = Metrics(model.num_classes, device, twelve_class_metrics=config['stage'] == 'finetune')
    optimizer.zero_grad(set_to_none=True)
    seen, window_seen, window_size = 0, 0, 0
    for images, labels in loader:
        if window_seen == 0:
            # Normalize a final partial accumulation window by its real image count.
            window_size = min(config['effective_batch'], len(loader.sampler) - seen)
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(images)
        # Keep ordinary CE for reporting and model selection. Label smoothing,
        # when configured for pretraining, changes only the training objective.
        ce = F.cross_entropy(logits.float(), labels, reduction='none')
        objective = (F.cross_entropy(logits.float(), labels, reduction='none',
                                     label_smoothing=config['label_smoothing'])
                     if config['label_smoothing'] else ce)
        (objective.sum() / window_size).backward()
        meter.update(logits, labels, ce, objective)
        seen += len(labels)
        window_seen += len(labels)
        if window_seen == window_size:
            if phase == 'finetune':
                parameters = [p for group in optimizer.param_groups
                              for p in group['params'] if p.grad is not None]
                torch.nn.utils.clip_grad_norm_(parameters, config['clip_norm'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_seen = 0
    return meter.result()


def make_scheduler(optimizer, config):
    """Reduce the learning rate after a plateau in the validation monitor."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler'])


def run(config_path, output, checkpoint=None, resume=False, device='cuda'):
    """Train a new run or restore its optimizer and random state from last.pt."""
    output = ROOT / output
    output.mkdir(parents=True, exist_ok=resume)
    config = json.loads((ROOT / config_path).read_text())
    saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False) if resume else None
    if saved is not None:
        config = saved['config']
        if saved['stop_reason']:
            return output / 'best.pt'
    setup(config['seed'], config['threads'])
    device = torch.device(device)
    pretrain = config['stage'] == 'pretrain'

    # Build the fixed datasets before constructing the model. ImageNet supplies
    # its class list; downstream labels always follow the twelve-class order.
    train_data, val_data = training_datasets(config)
    classes = train_data.classes if pretrain else NAMES
    train_loader = make_loader(train_data, config, True)
    val_loader = make_loader(val_data, config, False)
    model = MultiResolutionCNN(len(classes), config['seed'])
    if saved is not None:
        model.load_state_dict(saved['model'])
    elif checkpoint is not None:
        # Replace only the classifier entries with the new twelve-class head.
        # All convolutions, normalization parameters and BN statistics transfer.
        source = dict(torch.load(ROOT / checkpoint, map_location='cpu', weights_only=False)['model'])
        source.update({name: value for name, value in model.state_dict().items()
                       if name.startswith('classifier.')})
        model.load_state_dict(source)
    model = model.to(device, memory_format=torch.channels_last)

    # The head phase uses a separate optimizer. Full fine-tuning starts a new
    # AdamW optimizer with a lower rate for the pretrained feature extractor.
    head_epochs = config.get('head_epochs', 0)
    phase = saved['phase'] if saved else ('pretrain' if pretrain else
                                         'head' if head_epochs else 'finetune')
    set_phase(model, phase)
    optimizer = optimizer_for(model, config, phase)
    scheduler = None if phase == 'head' else make_scheduler(optimizer, config)
    state = dict(epoch=0, recent=[], reference_best=-float('inf'), bad_epochs=0,
                 last_lr_drop=None, lr_reductions=0, best_raw=-float('inf'),
                 best_ce=float('inf'), best_epoch=0)
    if saved is not None:
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler'])
        state = saved['state']
        restore_rng(saved['rng'], train_loader, val_loader, device)
    save_json(output / 'config.json', config)

    total_epochs = config['max_epochs'] + head_epochs
    for epoch in range(state['epoch'] + 1, total_epochs + 1):
        if not pretrain:
            train_loader.sampler.set_epoch(epoch)
        if not pretrain and epoch == head_epochs + 1:
            phase = 'finetune'
            optimizer = optimizer_for(model, config, phase)
            scheduler = make_scheduler(optimizer, config)
        if pretrain and epoch <= config['warmup_epochs']:
            # Warm up linearly from warmup_start_factor to the base SGD rate.
            factor = config['warmup_start_factor'] + (1 - config['warmup_start_factor']) * (epoch - 1) / (config['warmup_epochs'] - 1)
            optimizer.param_groups[0]['lr'] = config['lr'] * factor

        started = time.perf_counter()
        training = train_epoch(model, train_loader, optimizer, config, phase)
        validation = evaluate(model, val_loader, twelve_class_metrics=not pretrain)
        seconds = time.perf_counter() - started
        raw = validation['accuracy' if pretrain else 'macro_recall']

        # Select the checkpoint by the raw validation score. Use the mean of
        # three consecutive eligible scores for LR scheduling and early stopping.
        eligible = epoch > config['warmup_epochs'] if pretrain else phase == 'finetune'
        if eligible:
            state['recent'] = (state['recent'] + [raw])[-3:]
            if len(state['recent']) == 3:
                monitor = sum(state['recent']) / 3
                previous_lr = [group['lr'] for group in optimizer.param_groups]
                scheduler.step(monitor)
                if any(group['lr'] < old for group, old in zip(optimizer.param_groups, previous_lr)):
                    state['last_lr_drop'] = epoch
                    state['lr_reductions'] += 1
                if monitor > state['reference_best'] + config['improvement_threshold']:
                    state['reference_best'], state['bad_epochs'] = monitor, 0
                else:
                    state['bad_epochs'] += 1
        improved = raw > state['best_raw'] or (raw == state['best_raw'] and validation['ce'] < state['best_ce'])
        if improved:
            state.update(best_raw=raw, best_ce=validation['ce'], best_epoch=epoch)
        plateau = (epoch - head_epochs >= config['min_epochs']
                   and state['bad_epochs'] >= config['early_stop_patience']
                   and state['last_lr_drop'] is not None
                   and epoch - state['last_lr_drop'] >= config['epochs_after_lr_drop']
                   and state['lr_reductions'] >= config.get('minimum_lr_reductions', 0))
        reason = 'validation_plateau' if plateau else ('budget_limit' if epoch == total_epochs else None)
        state['epoch'] = epoch

        # best.pt contains the selected model. last.pt also contains everything
        # needed to continue optimization and image augmentation from this epoch.
        payload = dict(model=model.state_dict(), num_classes=len(classes), classes=classes,
                       config=config, phase=phase, epoch=epoch, validation=validation,
                       stop_reason=reason)
        if improved:
            torch.save(payload, output / 'best.pt')
        payload.update(optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict() if scheduler else None,
                       state=state, rng=save_rng(train_loader, val_loader, device))
        torch.save(payload, output / 'last.pt')
        save_json(output / 'summary.json', dict(state='complete' if reason else 'running',
                  completed_epochs=epoch, best_epoch=state['best_epoch'],
                  best_validation=state['best_raw'], stop_reason=reason))
        print(f"epoch={epoch} phase={phase} trainCE={training['ce']:.4f} "
              f"valCE={validation['ce']:.4f} valScore={raw:.4f} "
              f"best={state['best_epoch']} seconds={seconds:.1f}", flush=True)
        if reason:
            break
    return output / 'best.pt'


def test_checkpoint(checkpoint, output, device='cuda'):
    """Load the validation-selected model and evaluate the fixed test split once."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    setup(config['seed'], config['threads'])
    model = MultiResolutionCNN(saved['num_classes'], config['seed'])
    model.load_state_dict(saved['model'])
    model = model.to(device, memory_format=torch.channels_last)
    dataset = ClassificationData('test', config['data_root'], config['split_manifest'], config['input'])
    result = evaluate(model, make_loader(dataset, config, False))
    save_json(ROOT / output, result)
    return result
