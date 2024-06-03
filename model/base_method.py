import json
import os
import cv2
import torch
import logging
import numpy as np
import torch.nn as nn
import os.path as osp
import pytorch_lightning as pl
from timm.optim import Nadam, RAdam, AdamP, SGDP, Adafactor, Adahessian, RMSpropTF, NvNovoGrad, Lookahead
from timm.scheduler import TanhLRScheduler, StepLRScheduler, MultiStepLRScheduler
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None

optim_parameters = {
    'adam': {
        'weight_decay': 0,
    },
    'adamw': {
        'weight_decay': 0.01,
    },
    'sgd': {
        'momentum': 0,
        'dampening': 0,
        'weight_decay': 0,
        'nesterov': False,
    },
}


def print_log(message):
    print(message)
    logging.info(message)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def _threshold(x, y, t):
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred - true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred - true) ** 2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true) ** 2).mean()
    noise = ((true - pred) ** 2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def POD(hits, misses, eps=1e-6):
    """
    probability_of_detection
    Inputs:
    Outputs:
        pod = hits / (hits + misses) averaged over the T channels

    """
    pod = (hits + eps) / (hits + misses + eps)
    return np.mean(pod)


def SUCR(hits, fas, eps=1e-6):
    """
    success_rate
    Inputs:
    Outputs:
        sucr = hits / (hits + false_alarms) averaged over the D channels
    """
    sucr = (hits + eps) / (hits + fas + eps)
    return np.mean(sucr)


def CSI(hits, fas, misses, eps=1e-6):
    """
    critical_success_index
    Inputs:
    Outputs:
        csi = hits / (hits + false_alarms + misses) averaged over the D channels
    """
    csi = (hits + eps) / (hits + misses + fas + eps)
    return np.mean(csi)


def sevir_metrics(pred, true, threshold):
    """
    calcaulate t, p, hits, fas, misses
    Inputs:
    pred: [N, T, C, L, L]
    true: [N, T, C, L, L]
    threshold: float
    """
    pred = pred.transpose(1, 0, 2, 3, 4)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        channel_names (list | None): The name of different channels.
        spatial_norm (bool): Weather to normalize the metric by HxW.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i * c_width: (i + 1) * c_width, ...],
                                                     true[:, :, i * c_width: (i + 1) * c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i * c_width: (i + 1) * c_width, ...],
                                                     true[:, :, i * c_width: (i + 1) * c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i * c_width: (i + 1) * c_width, ...],
                                                       true[:, :, i * c_width: (i + 1) * c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses)

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'snr' in metrics:
        snr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr += SNR(pred[b, f], true[b, f])
        eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        lpips = 0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred = pred.transpose(0, 1, 3, 4, 2)
        true = true.transpose(0, 1, 3, 4, 2)
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                lpips += cal_lpips(pred[b, f], true[b, f])
        eval_res['lpips'] = lpips / (pred.shape[0] * pred.shape[1])

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def get_optim_scheduler(args, epoch, model, steps_per_epoch):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    # if weight_decay and filter_bias_and_bn:
    if args.filter_bias_and_bn:
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        else:
            skip = {}
        parameters = get_parameter_groups(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = optim_parameters.get(opt_lower, dict())
    opt_args.update(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    sched_lower = args.sched.lower()
    total_steps = epoch * steps_per_epoch
    by_epoch = True
    if sched_lower == 'onecycle':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps // len(args.gpus),
            final_div_factor=getattr(args, 'final_div_factor', 1e4))
        by_epoch = False
    elif sched_lower == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
    elif sched_lower == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch,
            t_in_epochs=True)  # update lr by_epoch
    elif sched_lower == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epoch,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch)
    elif sched_lower == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=args.decay_epoch,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch)
    else:
        assert False and "Invalid scheduler"

    return optimizer, lr_scheduler, by_epoch


class Base_method(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.metric_list = self.hparams.metrics
        self.channel_names = None
        self.spatial_norm = False
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams,
            self.hparams.epoch,
            self.model,
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            },
        }

    def forward(self, batch):
        NotImplementedError

    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x1, batch_x2, batch_y = batch
        pred_y = self(batch_x1, batch_x2)
        batch_y = batch_y[:, -pred_y.shape[1]:, :]
        loss = self.criterion(pred_y, batch_y)
        loss = torch.sqrt(loss)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x1, batch_x2, batch_y = batch
        pred_y = self(batch_x1, batch_x2)
        batch_y = batch_y[:, -pred_y.shape[1]:, :]
        outputs = {'inputs_x1': batch_x1.cpu().numpy(),  # 分别存储每部分数据
                   'inputs_x2': batch_x2.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)

        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
                                    self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list,
                                    channel_names=self.channel_names, spatial_norm=self.spatial_norm,
                                    threshold=self.hparams.get('metric_threshold', None))

        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs_x1', 'inputs_x2', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all
