# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR

# add by kevin.cao at 20.01.08
import torch.optim as optim
import numpy as np

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    
# add by kevin.cao at 20.01.08 =======



class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []

    def add(self, optimizer, lr, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)
        self.decay_every.append(decay_every)

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = self.step_lr(
                epoch_idx, init_lr, decay_every,
                0.1, optimizer)
                
    def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
        """Handles step decay of learning rate."""
        factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
        new_lr = base_lr * factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print('Set lr to ', new_lr)
        return optimizer
                
def make_PG_optimizer(cfg, model):

    # Masking will be done.
    # Checks.
    print(model)
    '''
    assert not args.lr and not args.lr_decay_every
    assert args.lr_mask and args.lr_mask_decay_every
    assert args.lr_classifier and args.lr_classifier_decay_every
    '''
    print('Performing masking.')
    '''
    for key, value in model.backbone.named_parameters():
        print(key, value)
    exit()
    '''
    optimizer_masks = optim.Adam(model.backbone.parameters(), lr=cfg.SOLVER.PG_MASK_LR)

    params = []
    lr = cfg.SOLVER.PG_HEAD_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    for key, value in model.rpn.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    for key, value in model.roi_heads.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_classifier = optim.Adam(params, lr=cfg.SOLVER.PG_HEAD_LR)

    optimizers = [optimizer_masks, optimizer_classifier]

    scheduler_masks = optim.lr_scheduler.MultiStepLR(optimizer_masks, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA)
    scheduler_classifier = optim.lr_scheduler.MultiStepLR(optimizer_classifier, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA)
    schedulers = [scheduler_masks, scheduler_classifier]
    
    return optimizers, schedulers
    
    
