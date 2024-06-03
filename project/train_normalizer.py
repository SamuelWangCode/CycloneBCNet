import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


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


def collect_data_for_normalization(value_dir, target_dir):
    all_inputs = []
    all_targets = []

    for file in os.listdir(value_dir):
        if file.endswith("_input.pt"):
            data = torch.load(os.path.join(value_dir, file))
            all_inputs.append(data.numpy())

    for file in os.listdir(target_dir):
        if file.endswith("_target.pt"):
            data = torch.load(os.path.join(target_dir, file))
            all_targets.append(data.numpy())

    return np.concatenate(all_inputs), np.concatenate(all_targets)


# 收集数据
value_dir = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
target_dir = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
inputs, targets = collect_data_for_normalization(value_dir, target_dir)
normalizer = MultiChannelNormalizer(num_channels=2)
normalizer.fit(np.vstack((inputs[:, 2:], targets[:, 2:])))  # 只对压强和风速归一化

# 保存归一化器
normalizer.save('./data_file/stats/', 'intensity')

normalizer.load('./data_file/stats/', 'intensity')
all_inputs = []
for file in os.listdir(value_dir):
    if file.endswith("_input.pt"):
        data = torch.load(os.path.join(value_dir, file))
        all_inputs.append(data.numpy())
a = np.concatenate(all_inputs)
a = a[:, 2:]
print(a)
a_verse = normalizer.transform(a)
print(a_verse)
a_reverse = normalizer.inverse_transform(a_verse)
print(a_reverse)


normalizer = MultiChannelNormalizer(num_channels=2)
normalizer.fit(torch.tensor([[100, 0], [180, 70]]))
normalizer.save('./data_file/stats/', 'position')
normalizer.load('./data_file/stats/', 'position')
all_inputs = []
for file in os.listdir(value_dir):
    if file.endswith("_input.pt"):
        data = torch.load(os.path.join(value_dir, file))
        all_inputs.append(data.numpy())
a = np.concatenate(all_inputs)
a = a[:, :2]
print(a)
a_verse = normalizer.transform(a)
print(a_verse)
a_reverse = normalizer.inverse_transform(a_verse)
print(a_reverse)


field_dir = '/data4/wxz_data/typhoon_intensity_bc/field_data_extraction'
all_inputs = []
for file in os.listdir(field_dir):
    if file.endswith('24.pt'):
        data = torch.load(os.path.join(field_dir, file))
        all_inputs.append(data)
all_data = np.concatenate(all_inputs, axis=0)
normalizer = FieldNormalizer(num_channels=73)
normalizer.fit(all_data)
normalizer.save('./data_file/stats/', 'field_24')
normalizer.load('./data_file/stats/', 'field_24')
print(all_data)
transformed_data = normalizer.transform(all_data)
print("Transformed Data:", transformed_data)
reversed_data = normalizer.inverse_transform(transformed_data)
print("Reversed Data:", reversed_data)


field_dir = '/data4/wxz_data/typhoon_intensity_bc/field_data_extraction'
all_inputs = []
for file in os.listdir(field_dir):
    if file.endswith('48.pt'):
        data = torch.load(os.path.join(field_dir, file))
        all_inputs.append(data)
all_data = np.concatenate(all_inputs, axis=0)
normalizer = FieldNormalizer(num_channels=73)
normalizer.fit(all_data)
normalizer.save('./data_file/stats/', 'field_48')


field_dir = '/data4/wxz_data/typhoon_intensity_bc/field_data_extraction'
all_inputs = []
for file in os.listdir(field_dir):
    if file.endswith('72.pt'):
        data = torch.load(os.path.join(field_dir, file))
        all_inputs.append(data)
a = np.concatenate(all_inputs)
normalizer = MultiChannelNormalizer(num_channels=73)
normalizer.fit(a)
all_data = np.concatenate(all_inputs, axis=0)
normalizer = FieldNormalizer(num_channels=73)
normalizer.fit(all_data)
normalizer.save('./data_file/stats/', 'field_72')

field_dir = '/data4/wxz_data/typhoon_intensity_bc/field_data_extraction'
all_inputs = []
for file in os.listdir(field_dir):
    if file.endswith('96.pt'):
        data = torch.load(os.path.join(field_dir, file))
        all_inputs.append(data)
a = np.concatenate(all_inputs)
normalizer = MultiChannelNormalizer(num_channels=73)
normalizer.fit(a)
all_data = np.concatenate(all_inputs, axis=0)
normalizer = FieldNormalizer(num_channels=73)
normalizer.fit(all_data)
normalizer.save('./data_file/stats/', 'field_96')
