#!/usr/bin/env python
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.utils.data
import wandb
from torch.utils.data.dataloader import default_collate

from src.datasets.definitions import MOD_RGB, MOD_SS_DENSE, MOD_ID
from src.datasets.transforms import get_transforms
from src.datasets.visualizations import compose
from src.utils.config import parse_config_and_args, convert_to_namespace, format_dict
from src.utils.helpers import verify_experiment_integrity, PersistentRandomSampler, SilentSummaryWriter, \
    tb_add_scalars, silent_torch_jit_trace_module, net_extract_modules_order, pretty_dict, get_zipped_size
from src.utils.module_surgery import net_prepare_training
from src.utils.resolvers import resolve_optimizer, resolve_semseg_dataset, resolve_semseg_model, PolyWarmupLR
from src.utils.tensorized_basis_modules_tr import TBasisWeightsFactoryProxy, TBasisWeightsFactory, TBasis


def semseg_compute_confusion(y_hat_lbl, y, C, semseg_ignore_class):
    assert torch.is_tensor(y_hat_lbl) and torch.is_tensor(y), 'Inputs must be torch tensors'
    assert y.device == y_hat_lbl.device, 'Input tensors have different device placement'

    if y_hat_lbl.dim() == 4 and y_hat_lbl.shape[1] == 1:
        y_hat_lbl = y_hat_lbl.squeeze(1)
    if y.dim() == 4 and y.shape[1] == 1:
        y = y.squeeze(1)

    mask = y != semseg_ignore_class
    y_hat_lbl = y_hat_lbl[mask]
    y = y[mask]

    # hack for bincounting 2 arrays together
    x = y_hat_lbl + C * y
    bincount_2d = torch.bincount(x.long(), minlength=C ** 2)
    assert bincount_2d.numel() == C ** 2, 'Internal error'
    conf = bincount_2d.view((C, C)).long()
    return conf


def semseg_accum_confusion_to_iou(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    iou_per_class = diag / (conf.sum(dim=1) + conf.sum(dim=0) - diag).clamp(min=1e-12)
    iou_mean = iou_per_class.mean()
    return iou_mean, iou_per_class


def visualize(tb, y_hat, rgb, rgb_tags, batch, dataset_train, cfg, step, tag):
    y_hat_lbl = y_hat.argmax(dim=1, keepdim=True)
    visualization_plan = []
    visualization_plan.append((MOD_RGB, rgb, rgb_tags))
    visualization_plan.append((MOD_SS_DENSE, y_hat_lbl, 'Prediction'))
    visualization_plan.append((MOD_SS_DENSE, batch[MOD_SS_DENSE], 'Ground truth'))
    vis = compose(
        visualization_plan, cfg,
        rgb_mean=dataset_train.rgb_mean,
        rgb_stddev=dataset_train.rgb_stddev,
        semseg_color_map=dataset_train.semseg_class_colors,
        semseg_ignore_class=dataset_train.ignore_label,
    )
    tb.add_image(tag, vis, global_step=step)


def main(cfg):
    assert cfg.experiment == 'semseg'

    seed = cfg.__dict__.get('random_seed', 2020)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    log_dir = cfg.log_dir
    tb_dir = os.path.join(log_dir, 'tb')
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(cfg.wandb_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path_latest = os.path.join(checkpoints_dir, 'checkpoint_latest.pth')
    checkpoint_path_best = os.path.join(checkpoints_dir, 'checkpoint_best.pth')
    is_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None

    # check the experiment is not resumed with different code and or settings
    verify_experiment_integrity(cfg)

    dataset_train, dataset_valid = resolve_semseg_dataset(cfg)

    semseg_ignore_class = dataset_train.ignore_label
    semseg_class_names = dataset_train.semseg_class_names
    semseg_num_classes = dataset_train.num_classes
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=semseg_ignore_class)

    model = resolve_semseg_model(cfg, semseg_num_classes)

    transforms_train = get_transforms(
        semseg_ignore_class=semseg_ignore_class,
        geom_scale_min=cfg.aug_geom_scale_min,
        geom_scale_max=cfg.aug_geom_scale_max,
        geom_tilt_max_deg=cfg.aug_geom_tilt_max_deg,
        geom_wiggle_max_ratio=cfg.aug_geom_wiggle_max_ratio,
        geom_reflect=cfg.aug_geom_reflect,
        crop_random=cfg.aug_input_crop_size,
        rgb_zero_mean_status=True,
        rgb_mean=dataset_train.rgb_mean,
        rgb_stddev=dataset_train.rgb_stddev,
    )

    transforms_valid = get_transforms(
        semseg_ignore_class=semseg_ignore_class,
        crop_for_passable=model.bottleneck_stride if not cfg.aug_geom_validation_center_crop_sts else 0,
        crop_center=cfg.aug_geom_validation_center_crop_size if cfg.aug_geom_validation_center_crop_sts else 0,
        rgb_zero_mean_status=True,
        rgb_mean=dataset_train.rgb_mean,
        rgb_stddev=dataset_train.rgb_stddev,
    )

    dataset_train.set_transforms(transforms_train)
    dataset_valid.set_transforms(transforms_valid)

    persistent_random_sampler = PersistentRandomSampler(
        dataset_train,
        cfg.num_training_steps * cfg.batch_size
    )

    classes_interest = (torch.nn.Conv2d, torch.nn.Linear)
    modules_order = net_extract_modules_order(model, (torch.randn(2, 3, 256, 256),), classes_interest)
    print('Modules order:' + '\n    '.join([''] + modules_order))

    weights_factory, weights_factory_proxy, weights_factory_jit, tbasis, compression_stats = (None,) * 5
    if cfg.compressed:
        weights_factory = TBasisWeightsFactory(
            cfg.basis_size,
            cfg.basis_rank,
            cfg.rank_adaptation,
            forward_batch_size=1,
        )
        weights_factory_proxy = TBasisWeightsFactoryProxy(weights_factory)
        tbasis = TBasis(
            cfg.basis_size, cfg.basis_rank, cfg.basis_mode, cfg.basis_trainable_sts, cfg.basis_init_seed,
        )
        if cfg.basis_pretrained_path is not None:
            state_dict = torch.load(os.path.join(cfg.root_models, cfg.basis_pretrained_path))
            state_dict.pop('last_basis', None)
            tbasis.load_state_dict(state_dict)
        compression_stats = net_prepare_training(model, tbasis, weights_factory_proxy, cfg)
        print('Compression: ', pretty_dict(compression_stats))
        print('JIT weights_factory...')
        basis = tbasis()
        weights_factory_jit = silent_torch_jit_trace_module(weights_factory, {'forward': (basis,)})
        del tbasis.last_basis
        print('JIT weights_factory done.')

    if is_cuda:
        model = model.cuda()
        if cfg.compressed:
            tbasis = tbasis.cuda()
            weights_factory = weights_factory.cuda()

    def forward_weights_factory():
        if cfg.compressed:
            basis = tbasis.forward()
            weights = weights_factory_jit.forward(basis)
            weights_factory_proxy.set_tbasis_tensors(weights)
            return weights
        return None

    optimizers = [
        resolve_optimizer(cfg.optimizer_model_name)(
            model.parameters(), **cfg.optimizer_model_kwargs
        )
    ]
    if cfg.compressed:
        optimizers.append(resolve_optimizer(cfg.optimizer_basis_name)(
            list(tbasis.parameters()) + list(weights_factory.parameters()),
            **cfg.optimizer_basis_kwargs
        ))
    lr_schedulers = [
        PolyWarmupLR(
            opt,
            power=cfg.lr_scheduler_power,
            num_steps=cfg.num_training_steps,
            num_warmup_steps=cfg.lr_warmup_steps if cfg.lr_warmup_sts else 0,
        ) for opt in optimizers
    ]

    step_loaded = 0
    metric_best = 0
    wandb_id = wandb.util.generate_id()

    # load persistent state
    if os.path.isfile(checkpoint_path_latest):
        state_dict = torch.load(checkpoint_path_latest)
        model.load_state_dict(state_dict['model'])
        if cfg.compressed:
            tbasis.load_state_dict(state_dict['tbasis'])
            weights_factory.load_state_dict(state_dict['weights_factory'])
        for i, (o, l) in enumerate(zip(optimizers, lr_schedulers)):
            o.load_state_dict(state_dict[f'optimizer_{i}'])
            l.load_state_dict(state_dict[f'lr_scheduler_{i}'])
        persistent_random_sampler.load_state_dict(state_dict['persistent_random_sampler'])
        step_loaded = state_dict['step']
        metric_best = state_dict['metric_best']
        persistent_random_sampler.fast_forward_to(step_loaded * cfg.batch_size)
        wandb_id = state_dict['wandb_id']
        if step_loaded == cfg.num_training_steps:
            print('Experiment was finished earlier; exiting')
            exit(0)

    # start dataloader workers
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=False,
        drop_last=True,
        sampler=persistent_random_sampler,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size_validation,
        num_workers=cfg.workers_validation,
        pin_memory=False,
        drop_last=False,
    )

    wandb.init(
        project=cfg.wandb_project,
        resume=True,
        name=cfg.experiment_name,
        id=wandb_id,
        config=cfg.__dict__,
        dir=cfg.wandb_dir,
        save_code=False,
    )
    wandb.tensorboard.patch(
        save=False,  # copies tb files into cloud and allows to run tensorboard in the cloud
        tensorboardX=False,
        pytorch=True,
    )

    # training loop preamble
    if step_loaded == 0:
        print('Started training')
    else:
        print(f'Resumed training from step {step_loaded}')
    time_start_sec = time.monotonic()
    iter_train = iter(dataloader_train)

    # training loop
    with SilentSummaryWriter(tb_dir) as tb:
        if cfg.compressed:
            tb_add_scalars(tb, 'compression', compression_stats)

        step = step_loaded
        while True:
            step += 1

            batch = next(iter_train)
            images = batch[MOD_RGB].cuda(non_blocking=True)
            target = batch[MOD_SS_DENSE].cuda(non_blocking=True)

            model.train()
            if cfg.compressed:
                weights_factory.train()
                tbasis.train()

            weights = forward_weights_factory()
            output = model(images)['y_hat_high']
            loss_ce = loss_fn(output, target.squeeze(1))
            if (loss_ce != loss_ce).int().sum().item() > 0:
                raise ValueError('Witnessed NaN loss - check LR and or gradients')
            loss = loss_ce

            loss_reg = 0.0
            if cfg.compressed and cfg.decompressed_L2_regularizer_coef > 0:
                num_el = sum([a.numel() for a in weights])
                loss_reg = sum([(a ** 2).sum() / num_el for a in weights])
                loss = loss + cfg.decompressed_L2_regularizer_coef * loss_reg

            [o.zero_grad() for o in optimizers]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                [l.step() for l in lr_schedulers]
            loss.backward()
            [o.step() for o in optimizers]

            if step % cfg.num_log_loss_steps == 0:
                if cfg.compressed:
                    tb_add_scalars(tb, 'batch', {
                        'loss': loss,
                        'loss_ce': loss_ce,
                        'loss_reg': loss_reg,
                    }, global_step=step)
                else:
                    tb.add_scalar('batch/loss', loss, global_step=step)
                tb_add_scalars(tb, 'progress', {
                    'lr': lr_schedulers[0].get_last_lr()[0],
                    'pct_done': 100 * step / cfg.num_training_steps,
                    'eta_hrs': (time.monotonic() - time_start_sec) * (cfg.num_training_steps - step) /
                               ((step - step_loaded) * 3600)
                }, global_step=step)

            if step % cfg.num_validation_steps == 0:
                model.eval()

                with torch.no_grad():
                    rgb_tags = [f'{dataset_train.index2name(id.item())}' for id in batch[MOD_ID]]
                    visualize(tb, output, images, rgb_tags, batch, dataset_train, cfg, step, 'batch')
                    for ds, ids, prefix in (
                            (dataset_train, cfg.observe_train_ids, 'train'),
                            (dataset_valid, cfg.observe_valid_ids, 'valid'),
                    ):
                        list_samples = []
                        for i in ids:
                            if type(i) is str:
                                i = ds.name2index(i)
                            list_samples.append(ds.get(i, override_transforms=transforms_valid))
                        list_prefix = (f'{prefix}/',) * len(ids)
                        batch = default_collate(list_samples)
                        images = batch[MOD_RGB]
                        output = model(images.cuda())['y_hat_high'].cpu()
                        rgb_tags = [f'{prefix}{ds.index2name(id.item())}' for prefix, id in zip(list_prefix, batch[MOD_ID])]
                        visualize(tb, output, images, rgb_tags, batch, dataset_train, cfg, step, 'fixed')

                    if cfg.compressed:
                        for k, v in model.state_dict().items():
                            if 'running_var' in k:
                                tb.add_histogram(f'stats_bn/{k}', v, global_step=step, bins=64)
                        tb.add_histogram(f'stats_factory/basis_values', tbasis.last_basis, global_step=step, bins=64)
                        tb.add_histogram(f'stats_factory/coef_values', weights_factory.weights, global_step=step, bins=64)
                        tb.add_scalar('stats_factory/basis_std', tbasis.last_basis.std().item(), global_step=step)
                        tb.add_scalar('stats_factory/coef_std', weights_factory.weights.std().item(), global_step=step)
                        szzip_without_basis = get_zipped_size({
                            'n': model.state_dict(),
                            'c': weights_factory.state_dict(),
                        })
                        szzip_with_basis = get_zipped_size({
                            'n': model.state_dict(),
                            'c': weights_factory.state_dict(),
                            'b': tbasis.state_dict(),
                        })
                        tb_add_scalars(tb, 'compression', {
                            'size_zip_without_basis': szzip_without_basis,
                            'size_zip_with_basis': szzip_with_basis,
                        }, global_step=step)
                    else:
                        szzip_original_total = get_zipped_size(model.state_dict())
                        tb.add_scalar('compression/szzip_original_total', szzip_original_total, global_step=step)

                    confusion = None
                    for batch in dataloader_valid:
                        images = batch[MOD_RGB].cuda(non_blocking=True)
                        target = batch[MOD_SS_DENSE].cuda(non_blocking=True)
                        output = model(images)['y_hat_high']
                        output = output.argmax(dim=1, keepdim=True)
                        conf = semseg_compute_confusion(output, target, semseg_num_classes, semseg_ignore_class)
                        confusion = conf if confusion is None else confusion + conf
                    iou_mean, iou_per_class = semseg_accum_confusion_to_iou(confusion)
                    metrics = {semseg_class_names[i]: iou for i, iou in enumerate(iou_per_class)}

                metric_new_best = max(iou_mean, metric_best)
                tb.add_scalar('metrics/iou_mean', iou_mean, global_step=step)
                tb_add_scalars(tb, 'metrics', metrics, global_step=step)
                tb.add_scalar('metrics_best/iou_mean', metric_new_best, global_step=step)

                have_new_best = metric_best != metric_new_best
                if have_new_best:
                    print(f'Step {step}: iou_mean improved from {metric_best} to {metric_new_best}')
                    metric_best = metric_new_best
                else:
                    print(f'Step {step}: iou_mean did not improve')

                # prepare checkpoint
                state_dict = {
                    'model': model.state_dict(),
                    'persistent_random_sampler': persistent_random_sampler.state_dict(),
                    'step': step,
                    'metric_best': metric_best,
                    'wandb_id': wandb_id,
                }
                for i, (o, l) in enumerate(zip(optimizers, lr_schedulers)):
                    state_dict.update({
                        f'optimizer_{i}': o.state_dict(),
                        f'lr_scheduler_{i}': l.state_dict(),
                    })
                if cfg.compressed:
                    state_dict['tbasis'] = tbasis.state_dict()
                    state_dict['weights_factory'] = weights_factory.state_dict()
                torch.save(state_dict, checkpoint_path_latest + '.tmp')

                # handle best model artifacts
                if have_new_best:
                    # copy current checkpoint to best model checkpoint
                    shutil.copy(checkpoint_path_latest + '.tmp', checkpoint_path_best + '.tmp')
                    # commit best artifacts
                    os.rename(checkpoint_path_best + '.tmp', checkpoint_path_best)

                # commit checkpoint
                os.rename(checkpoint_path_latest + '.tmp', checkpoint_path_latest)

            if step == cfg.num_training_steps:
                break

    print(f'Step {step}: finished training')


if __name__ == '__main__':
    cfg, cfg_warnings = parse_config_and_args()
    cfg = convert_to_namespace(cfg)
    if len(cfg_warnings) > 0:
        print('\n'.join(cfg_warnings))
    print(format_dict(cfg.__dict__))
    main(cfg)
