import math

import pytorch_lightning as pl

from typhoon_intensity_bc.model.bcnet import BCNet
from typhoon_intensity_bc.project.construct_dataset import get_dataloader, position_normalizer, \
    test_csv_file_24, root_dir_field, root_dir_value, root_dir_target, field_normalizer_24, intensity_normalizer, \
    root_dir_track

model_params = {
    'save_dir': './data_file/BCNet/typhoon_intensity_24h',
    'lr': 0.00006,
    'opt': 'adamw',  # 选择优化器类型
    'weight_decay': 0.01,  # 设置权重衰减
    'filter_bias_and_bn': True,  # 是否过滤偏置和批归一化层的权重
    'pre_seq_length': 5,
    'aft_seq_length': 5,
    'test_mean': None,
    'test_std': None,
    'model_config': {
        # 模型配置参数
        'in_shape1': (5, 73, 53, 53),
        'in_shape2': (5, 4),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'output_dim': 2,
        'model_type': 'gSTA',
        'mlp_ratio': 8.,
        'drop': 0.0,
        'drop_path': 0.0,
        'spatio_kernel_enc': 3,
        'act_inplace': True,
        'center_weight': 2.0,
        'N': 13
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 2,
    'steps_per_epoch': math.ceil(1408 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}

trainer_model = BCNet(**model_params)
# 初始化 PyTorch Lightning 训练器
trainer = pl.Trainer(max_epochs=10000, accelerator="gpu",
                     devices=[4],
                     # devices=[0],
                     log_every_n_steps=1,
                     enable_checkpointing=True,
                     enable_progress_bar=True,
                     )

test_loader = get_dataloader(test_csv_file_24, model_params['batch_size'], root_dir_field, root_dir_value, root_dir_track,
                             root_dir_target, shuffle=False,
                             field_normalizer=field_normalizer_24, intensity_normalizer=intensity_normalizer,
                             position_normalizer=position_normalizer)
trainer.test(trainer_model, dataloaders=test_loader,
             ckpt_path='./data_file/BCNet/typhoon_track_24h/epoch=3534-step=7070.ckpt')
