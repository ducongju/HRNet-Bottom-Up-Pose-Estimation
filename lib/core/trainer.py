# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from utils.utils import AverageMeter
from utils.vis import save_debug_images


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict):
    logger = logging.getLogger("Training")

    # 计算并存储平均值和当前值
    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    offset_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    model.train()

    end = time.time()
    for i, (images, heatmaps, masks, offsets, weights) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs, poffsets = model(images)  # 前向传播

        heatmaps = [list(map(lambda x: x.cuda(non_blocking=True), heatmap))
                    for heatmap in heatmaps]
        masks = [list(map(lambda x: x.cuda(non_blocking=True), mask))
                 for mask in masks]
        offsets = [list(map(lambda x: x.cuda(non_blocking=True), offset))
                   for offset in offsets]
        offset_weights = [
            list(map(lambda x: x.cuda(non_blocking=True), weight)) for weight in weights]

        heatmaps_losses, offset_losses = \
            loss_factory(outputs, poffsets, heatmaps,
                         masks, offsets, offset_weights)  # 计算loss

        loss = 0

        # 如果输出的loss有两个分辨率, 将他们取平均
        for idx in range(cfg.LOSS.NUM_STAGES):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                heatmaps_loss_meter[idx].update(
                    heatmaps_loss.item(), images.size(0)
                )
                loss = loss + heatmaps_loss
            if offset_losses[idx] is not None:
                offset_loss = offset_losses[idx]
                offset_loss_meter[idx].update(
                    offset_loss.item(), images.size(0)
                )
                loss = loss + offset_loss

        optimizer.zero_grad()  # 梯度置零
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        batch_time.update(time.time() - end)
        end = time.time()

        # Time: 一个batch耗费的时间, 一个batch耗费的平均时间
        # Speed: 一个GPU每秒处理的sample个数
        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(
                          heatmaps_loss_meter, 'heatmaps'),
                      offset_loss=_get_loss_info(offset_loss_meter, 'offset')
                  )
            logger.info(msg)

            writer = writer_dict['writer']  # 设置tensorboard
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                # 添加用于可视化的数据
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(i),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_offset_loss'.format(idx),
                    offset_loss_meter[idx].val,
                    global_steps
                )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            for scale_idx in range(len(cfg.DATASET.OUTPUT_SIZE)):
                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx]
                )
                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx],
                    outputs[scale_idx], prefix_scale
                )


def _get_loss_info(loss_meters, loss_name):
    msg = ''
    for i, meter in enumerate(loss_meters):
        msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            i=i, name=loss_name, meter=meter
        )

    return msg
