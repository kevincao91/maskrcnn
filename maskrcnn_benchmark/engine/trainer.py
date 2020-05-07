# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    # add by kevin.cao at 20.01.08 ===
    
    print(model)
    
    '''
    # add by kevim.cao at 19.12.19 ===
    for key, value in model.named_parameters():
        print(key, value.requires_grad)
    # =================
    '''

    PGflag = True if isinstance(optimizer,list) else False
    #if self.args.threshold_fn == 'binarizer':
    '''
    if True:
        print('Num 0ed out parameters:')
        for idx, module in enumerate(model.backbone.body.modules()):
            if 'ElementWise' in str(type(module)):
                num_zero = module.mask_real.data.lt(5e-3).sum()
                total = module.mask_real.data.numel()
                print(idx, num_zero, total)
    print('-' * 20)
    '''
    # ================================
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        #print('read img num:', len(targets))

        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # add by kevin.cao at 20.01.08 ===
        if PGflag:
            for scheduler_ in scheduler:
                scheduler_.step()
        else:
            scheduler.step()
            
        model.zero_grad()
        # ================================
        '''
        # ===
        print('1==' * 20, iteration)
        if True:
            print('Num 0ed out parameters:')
            for idx, module in enumerate(model.backbone.body.modules()):
                if 'ElementWise' in str(type(module)):
                    print(module.mask_real[0][0])
                    num_zero = module.mask_real.data.lt(5e-3).sum()
                    total = module.mask_real.data.numel()
                    print(idx, num_zero, total)
                if idx >= 1:
                    break
        print('-' * 20)
        # =========
        '''

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # add by kevin.cao at 20.01.08 ===
        if PGflag:
            for optimizer_ in optimizer:
                optimizer_.zero_grad()
        else:
            optimizer.zero_grad()
        # ================================
        '''
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        '''
        
        losses.backward()

        '''
        # ===
        print('3==' * 20, iteration)
        if True:
            print('Num 0ed out parameters:')
            for idx, module in enumerate(model.backbone.body.modules()):
                if 'ElementWise' in str(type(module)):
                    print(module.mask_real[0][0])
                    num_zero = module.mask_real.data.lt(5e-3).sum()
                    total = module.mask_real.data.numel()
                    print(idx, num_zero, total)
                if idx >= 1:
                    break
        print('-' * 20)
        # =========
        '''
        
        # add by kevin.cao at 20.01.08 ===
        if PGflag:
            for optimizer_ in optimizer:
                optimizer_.step()
        else:
            optimizer.step()
        # ================================
        # ===
        '''
        print('4==' * 20, iteration)
        if True:
            print('Num 0ed out parameters:')
            for idx, module in enumerate(model.backbone.body.modules()):
                if 'ElementWise' in str(type(module)):
                    print(module.mask_real[0][0])
                    num_zero = module.mask_real.data.lt(5e-3).sum()
                    total = module.mask_real.data.numel()
                    print(idx, num_zero, total)
                    print (module.mask_real[0][0].grad_fn)
                if idx >= 1:
                    break
        print('-' * 20)
        # =========
        '''
        '''
        if iteration == 50:
            exit() 
        '''
        
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            # add by kevin.cao at 20.01.08 ===
            if PGflag:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr_mask: {lr_mask:.6f}",
                            "lr_head: {lr_head:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_mask=optimizer[0].param_groups[0]["lr"],
                        lr_head=optimizer[1].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            else:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            # ================================
        
        #if self.args.threshold_fn == 'binarizer':
        '''
        if iteration % 20 == 0:
            print('Num 0ed out parameters:')
            for idx, module in enumerate(model.backbone.body.modules()):
                if 'ElementWise' in str(type(module)):
                    if idx<=1:
                        print(module.mask_real[0][0])
                    num_zero = module.mask_real.data.lt(5e-3).sum()
                    total = module.mask_real.data.numel()
                    print(idx, num_zero, total)
            print('-' * 20)
        '''
        # ================================
        
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
