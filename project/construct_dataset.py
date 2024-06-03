import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import os

from torch.utils.data import DataLoader, Dataset

from typhoon_intensity_bc.model.model import BCModel

# track 24h
model_params_track_24 = {
    'save_dir': './data_file/BCNet/typhoon_track_24h',
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
        'in_shape2': (5, 2),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'output_dim': 2,
        'model_type': 'gSTA',
        'mlp_ratio': 8.,
        'drop': 0.03,
        'drop_path': 0.0,
        'spatio_kernel_enc': 3,
        'act_inplace': True,
        'center_weight': 1.0,
        'N': 13
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 10,
    'steps_per_epoch': math.ceil(1408 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}

# track 48h
model_params_track_48 = {
    'save_dir': './data_file/BCNet/typhoon_track_48h',
    'lr': 0.00006,
    'opt': 'adamw',  # 选择优化器类型
    'weight_decay': 0.01,  # 设置权重衰减
    'filter_bias_and_bn': True,  # 是否过滤偏置和批归一化层的权重
    'pre_seq_length': 9,
    'aft_seq_length': 9,
    'test_mean': None,
    'test_std': None,
    'model_config': {
        # 模型配置参数
        'in_shape1': (9, 73, 57, 57),
        'in_shape2': (9, 2),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'output_dim': 2,
        'model_type': 'gSTA',
        'mlp_ratio': 8.,
        'drop': 0.03,
        'drop_path': 0.0,
        'spatio_kernel_enc': 3,
        'act_inplace': True,
        'center_weight': 1.0,
        'N': 17
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 10,
    'steps_per_epoch': math.ceil(1164 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}

# track 72h
model_params_track_72 = {
    'save_dir': './data_file/BCNet/typhoon_track_72h',
    'lr': 0.00006,
    'opt': 'adamw',  # 选择优化器类型
    'weight_decay': 0.01,  # 设置权重衰减
    'filter_bias_and_bn': True,  # 是否过滤偏置和批归一化层的权重
    'pre_seq_length': 13,
    'aft_seq_length': 13,
    'test_mean': None,
    'test_std': None,
    'model_config': {
        # 模型配置参数
        'in_shape1': (13, 73, 65, 65),
        'in_shape2': (13, 2),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'output_dim': 2,
        'model_type': 'gSTA',
        'mlp_ratio': 8.,
        'drop': 0.03,
        'drop_path': 0.0,
        'spatio_kernel_enc': 3,
        'act_inplace': True,
        'center_weight': 1.0,
        'N': 25
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 10,
    'steps_per_epoch': math.ceil(928 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}

# track 96h
model_params_track_96 = {
    'save_dir': './data_file/BCNet/typhoon_track_96h',
    'lr': 0.00006,
    'opt': 'adamw',  # 选择优化器类型
    'weight_decay': 0.01,  # 设置权重衰减
    'filter_bias_and_bn': True,  # 是否过滤偏置和批归一化层的权重
    'pre_seq_length': 17,
    'aft_seq_length': 17,
    'test_mean': None,
    'test_std': None,
    'model_config': {
        # 模型配置参数
        'in_shape1': (17, 73, 73, 73),
        'in_shape2': (17, 2),
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
        'center_weight': 1.0,
        'N': 33
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 2,
    'steps_per_epoch': math.ceil(720 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}

# intensity 24h
model_params_24 = {
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

# intensity 48h
model_params_48 = {
    'save_dir': './data_file/BCNet/typhoon_intensity_48h',
    'lr': 0.00006,
    'opt': 'adamw',  # 选择优化器类型
    'weight_decay': 0.01,  # 设置权重衰减
    'filter_bias_and_bn': True,  # 是否过滤偏置和批归一化层的权重
    'pre_seq_length': 9,
    'aft_seq_length': 9,
    'test_mean': None,
    'test_std': None,
    'model_config': {
        # 模型配置参数
        'in_shape1': (9, 73, 57, 57),
        'in_shape2': (9, 4),
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
        'N': 17
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 2,
    'steps_per_epoch': math.ceil(1164 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}

# intensity 72h
model_params_72 = {
    'save_dir': './data_file/BCNet/typhoon_intensity_72h',
    'lr': 0.00006,
    'opt': 'adamw',  # 选择优化器类型
    'weight_decay': 0.01,  # 设置权重衰减
    'filter_bias_and_bn': True,  # 是否过滤偏置和批归一化层的权重
    'pre_seq_length': 13,
    'aft_seq_length': 13,
    'test_mean': None,
    'test_std': None,
    'model_config': {
        # 模型配置参数
        'in_shape1': (13, 73, 65, 65),
        'in_shape2': (13, 4),
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
        'N': 25
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 2,
    'steps_per_epoch': math.ceil(928 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}

# intensity 96h
model_params_96 = {
    'save_dir': './data_file/BCNet/typhoon_intensity_96h',
    'lr': 0.00006,
    'opt': 'adamw',  # 选择优化器类型
    'weight_decay': 0.01,  # 设置权重衰减
    'filter_bias_and_bn': True,  # 是否过滤偏置和批归一化层的权重
    'pre_seq_length': 17,
    'aft_seq_length': 17,
    'test_mean': None,
    'test_std': None,
    'model_config': {
        # 模型配置参数
        'in_shape1': (17, 73, 73, 73),
        'in_shape2': (17, 4),
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
        'N': 33
    },
    'metrics': ['mse', 'mae', 'rmse'],
    'epoch': 10000,
    'batch_size': 256,
    'num_batches': 2,
    'steps_per_epoch': math.ceil(720 // 256),
    'sched': 'cosine',  # 调度器类型
    'min_lr': 0.000001,  # 最小学习率
    'warmup_lr': 0.0005,  # 热身学习率
    'warmup_epoch': 2,  # 热身周期数
    'decay_epoch': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000],
    # 衰减周期
    'decay_rate': 0.1,  # 衰减率
}


def load_model(model_path, device, model_params):
    model = BCModel(**model_params['model_config'])
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # 创建一个新的状态字典，去除前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")  # 去除前缀
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


class MultiChannelNormalizer:
    def __init__(self, num_channels):
        self.scalers = {i: MinMaxScaler() for i in range(num_channels)}

    def fit(self, data):
        for i in range(data.shape[1]):  # 假设数据的形状为 (samples, channels, ...)
            self.scalers[i].fit(data[:, i].reshape(-1, 1))

    def transform(self, data):
        transformed_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            transformed_data[:, i] = self.scalers[i].transform(data[:, i].reshape(-1, 1)).flatten()
        return transformed_data

    def inverse_transform(self, data):
        inv_transformed_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            inv_transformed_data[:, i] = self.scalers[i].inverse_transform(data[:, i].reshape(-1, 1)).flatten()
        return inv_transformed_data

    def save(self, path, name):
        torch.save(self.scalers, os.path.join(path, f'{name}.pt'))

    def load(self, path, name):
        self.scalers = torch.load(os.path.join(path, f'{name}.pt'))


class FieldNormalizer:
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.scalers = {i: MinMaxScaler() for i in range(num_channels)}

    def fit(self, data):
        # Expect data to be in shape [N, T, C, W, H]
        for c in range(self.num_channels):
            channel_data = data[:, c, :, :].reshape(-1, 1)  # Flatten the channel data
            self.scalers[c].fit(channel_data)

    def transform(self, data):
        # Transform data
        transformed = np.empty_like(data)
        for c in range(self.num_channels):
            original_shape = data[:, c, :, :].shape
            channel_data = data[:, c, :, :].reshape(-1, 1)
            transformed[:, c, :, :] = self.scalers[c].transform(channel_data).reshape(original_shape)
        return transformed

    def inverse_transform(self, data):
        # Inverse transform data
        inv_transformed = np.empty_like(data)
        for c in range(self.num_channels):
            original_shape = data[:, c, :, :].shape
            channel_data = data[:, c, :, :].reshape(-1, 1)
            inv_transformed[:, c, :, :] = self.scalers[c].inverse_transform(channel_data).reshape(original_shape)
        return inv_transformed

    def save(self, path, name):
        # Save all scalers in one file
        torch.save(self.scalers, os.path.join(path, f'{name}.pt'))

    def load(self, path, name):
        # Load all scalers from one file
        self.scalers = torch.load(os.path.join(path, f'{name}.pt'))


class TyphoonDataset(Dataset):
    def __init__(self, csv_file, root_dir_field, root_dir_value, root_dir_track, root_dir_target, track_model=None,
                 field_normalizer=None,
                 intensity_normalizer=None, position_normalizer=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir_field = root_dir_field
        self.root_dir_value = root_dir_value
        self.root_dir_track = root_dir_track
        self.root_dir_target = root_dir_target
        self.track_model = track_model
        self.field_normalizer = field_normalizer
        self.intensity_normalizer = intensity_normalizer
        self.position_normalizer = position_normalizer

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        field_path = os.path.join(self.root_dir_field, f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}.pt")
        value_input_path = os.path.join(self.root_dir_value,
                                        f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_input.pt")
        track_input_path = os.path.join(self.root_dir_track,
                                        f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}.pt")
        target_path = os.path.join(self.root_dir_target,
                                   f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_target.pt")
        field_data = torch.load(field_path)
        value_input = torch.load(value_input_path)
        track_input = torch.load(track_input_path)
        target = torch.load(target_path)
        field_input = torch.tensor(self.field_normalizer.transform(field_data))
        # track
        value_input = torch.tensor(self.position_normalizer.transform(value_input[:, :2]))
        target = torch.tensor(self.position_normalizer.transform(target[:, :2]))
        # intensity
        # value_input = torch.tensor(self.intensity_normalizer.transform(value_input[:, 2:]))
        # track_input = torch.tensor(self.position_normalizer.transform(track_input[:, :]))
        # target = torch.tensor(self.intensity_normalizer.transform(target[:, 2:]))
        # value_input = torch.cat((track_input, value_input), dim=1)

        return field_input, value_input, target


# 数据加载器
def get_dataloader(csv_file, batch_size, root_dir_field, root_dir_value, root_dir_track, root_dir_target, shuffle=True,
                   track_model=None, field_normalizer=None, intensity_normalizer=None, position_normalizer=None):
    dataset = TyphoonDataset(csv_file=csv_file, root_dir_field=root_dir_field, root_dir_value=root_dir_value,
                             root_dir_track=root_dir_track, root_dir_target=root_dir_target,
                             field_normalizer=field_normalizer,
                             intensity_normalizer=intensity_normalizer, position_normalizer=position_normalizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)


# track
# track_model = None
# intensity 24h
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# track_model = load_model('./data_file/BCNet/typhoon_track_24h/epoch=297-step=596.ckpt', device, model_params_track)

# 示例路径和参数
train_csv_file_24 = './data_file/forecast_24h_train_set.csv'
valid_csv_file_24 = './data_file/forecast_24h_valid_set.csv'
test_csv_file_24 = './data_file/forecast_24h_test_set.csv'
train_csv_file_48 = './data_file/forecast_48h_train_set.csv'
valid_csv_file_48 = './data_file/forecast_48h_valid_set.csv'
test_csv_file_48 = './data_file/forecast_48h_test_set.csv'
train_csv_file_72 = './data_file/forecast_72h_train_set.csv'
valid_csv_file_72 = './data_file/forecast_72h_valid_set.csv'
test_csv_file_72 = './data_file/forecast_72h_test_set.csv'
train_csv_file_96 = './data_file/forecast_96h_train_set.csv'
valid_csv_file_96 = './data_file/forecast_96h_valid_set.csv'
test_csv_file_96 = './data_file/forecast_96h_test_set.csv'
root_dir_field = '/data4/wxz_data/typhoon_intensity_bc/field_data_extraction'
root_dir_value = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
root_dir_track = '/data4/wxz_data/typhoon_intensity_bc/track_forecast_data'
root_dir_target = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'

# 初始化归一化器并加载数据
position_normalizer = MultiChannelNormalizer(num_channels=2)
position_normalizer.load('./data_file/stats', 'position')
intensity_normalizer = MultiChannelNormalizer(num_channels=2)
intensity_normalizer.load('./data_file/stats', 'intensity')
field_normalizer_24 = FieldNormalizer(num_channels=73)
field_normalizer_24.load('./data_file/stats', 'field_24')
field_normalizer_48 = FieldNormalizer(num_channels=73)
field_normalizer_48.load('./data_file/stats', 'field_48')
field_normalizer_72 = FieldNormalizer(num_channels=73)
field_normalizer_72.load('./data_file/stats', 'field_72')
field_normalizer_96 = FieldNormalizer(num_channels=73)
field_normalizer_96.load('./data_file/stats', 'field_96')

# 24h
# train_loader = get_dataloader(train_csv_file_24, model_params_24['batch_size'], root_dir_field, root_dir_value,
#                               root_dir_track, root_dir_target, shuffle=True,
#                               field_normalizer=field_normalizer_24, intensity_normalizer=intensity_normalizer,
#                               position_normalizer=position_normalizer)
# val_loader = get_dataloader(valid_csv_file_24, model_params_24['batch_size'], root_dir_field, root_dir_value,
#                             root_dir_track, root_dir_target, shuffle=False,
#                             field_normalizer=field_normalizer_24, intensity_normalizer=intensity_normalizer,
#                             position_normalizer=position_normalizer)
# test_loader = get_dataloader(test_csv_file_24, model_params_24['batch_size'], root_dir_field, root_dir_value,
#                              root_dir_track, root_dir_target, shuffle=False,
#                              field_normalizer=field_normalizer_24, intensity_normalizer=intensity_normalizer,
#                              position_normalizer=position_normalizer)

# 48h
# train_loader = get_dataloader(train_csv_file_48, model_params_48['batch_size'], root_dir_field, root_dir_value,
#                               root_dir_track, root_dir_target, shuffle=True,
#                               field_normalizer=field_normalizer_48, intensity_normalizer=intensity_normalizer,
#                               position_normalizer=position_normalizer)
# val_loader = get_dataloader(valid_csv_file_48, model_params_48['batch_size'], root_dir_field, root_dir_value,
#                             root_dir_track, root_dir_target, shuffle=False,
#                             field_normalizer=field_normalizer_48, intensity_normalizer=intensity_normalizer,
#                             position_normalizer=position_normalizer)
# test_loader = get_dataloader(test_csv_file_48, model_params_48['batch_size'], root_dir_field, root_dir_value,
#                              root_dir_track, root_dir_target, shuffle=False,
#                              field_normalizer=field_normalizer_48, intensity_normalizer=intensity_normalizer,
#                              position_normalizer=position_normalizer)

# 72h
# train_loader = get_dataloader(train_csv_file_72, model_params_72['batch_size'], root_dir_field, root_dir_value,
#                               root_dir_track, root_dir_target, shuffle=True,
#                               field_normalizer=field_normalizer_72, intensity_normalizer=intensity_normalizer,
#                               position_normalizer=position_normalizer)
# val_loader = get_dataloader(valid_csv_file_72, model_params_72['batch_size'], root_dir_field, root_dir_value,
#                             root_dir_track, root_dir_target, shuffle=False,
#                             field_normalizer=field_normalizer_72, intensity_normalizer=intensity_normalizer,
#                             position_normalizer=position_normalizer)
# test_loader = get_dataloader(test_csv_file_72, model_params_72['batch_size'], root_dir_field, root_dir_value,
#                              root_dir_track, root_dir_target, shuffle=False,
#                              field_normalizer=field_normalizer_72, intensity_normalizer=intensity_normalizer,
#                              position_normalizer=position_normalizer)

# 96h
train_loader = get_dataloader(train_csv_file_96, model_params_96['batch_size'], root_dir_field, root_dir_value,
                              root_dir_track, root_dir_target, shuffle=True,
                              field_normalizer=field_normalizer_96, intensity_normalizer=intensity_normalizer,
                              position_normalizer=position_normalizer)
val_loader = get_dataloader(valid_csv_file_96, model_params_96['batch_size'], root_dir_field, root_dir_value,
                            root_dir_track, root_dir_target, shuffle=False,
                            field_normalizer=field_normalizer_96, intensity_normalizer=intensity_normalizer,
                            position_normalizer=position_normalizer)
test_loader = get_dataloader(test_csv_file_96, model_params_96['batch_size'], root_dir_field, root_dir_value,
                             root_dir_track, root_dir_target, shuffle=False,
                             field_normalizer=field_normalizer_96, intensity_normalizer=intensity_normalizer,
                             position_normalizer=position_normalizer)
