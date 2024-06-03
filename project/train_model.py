import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from typhoon_intensity_bc.model.bcnet import BCNet
from typhoon_intensity_bc.project.construct_dataset import train_loader, val_loader, test_loader, model_params_24, \
    model_params_48, model_params_72, model_params_96, model_params_track_24, \
    model_params_track_48, model_params_track_72, model_params_track_96

# forecast_hour = 24
# forecast_hour = 48
# forecast_hour = 72
forecast_hour = 96
type = 'track'
# type = 'intensity'
checkpoint_callback = ModelCheckpoint(
    dirpath=f'./data_file/BCNet/typhoon_{type}_{forecast_hour}h/',  # 指定保存模型的路径
    save_top_k=3,  # 保存表现最好的3个模型
    verbose=True,
    monitor='val_loss',  # 指定监视的验证损失
    mode='min',  # “min”模式表示损失越小越好
    every_n_epochs=1,  # 每个epoch保存一次
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=200,  # 设定100个epoch无改进则停止
    verbose=True,
    mode='min'
)

# 实例化训练模块
if type == 'intensity':
    if forecast_hour == 24:
        model_params = model_params_24
    elif forecast_hour == 48:
        model_params = model_params_48
    elif forecast_hour == 72:
        model_params = model_params_72
    else:
        model_params = model_params_96
    trainer_model = BCNet(**model_params)
else:
    if forecast_hour == 24:
        model_params_track = model_params_track_24
    elif forecast_hour == 48:
        model_params_track = model_params_track_48
    elif forecast_hour == 72:
        model_params_track = model_params_track_72
    else:
        model_params_track = model_params_track_96
    trainer_model = BCNet(**model_params_track)
strategy = DDPStrategy(find_unused_parameters=True)
# 初始化 PyTorch Lightning 训练器
trainer = pl.Trainer(max_epochs=10000, strategy=strategy, accelerator="gpu",
                     devices=[5, 6, 7],
                     # devices=[0],
                     log_every_n_steps=1,
                     enable_checkpointing=True,
                     enable_progress_bar=True,
                     callbacks=[checkpoint_callback, early_stop_callback],
                     )
trainer.fit(trainer_model, train_loader, val_loader,
            ckpt_path=f'./data_file/BCNet/typhoon_{type}_96h/epoch=3031-step=3032.ckpt'
            )
trainer.test(dataloaders=test_loader)
