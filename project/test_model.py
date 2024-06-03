import os
import pandas as pd
import numpy as np
from haversine import haversine
import torch

from typhoon_intensity_bc.project.construct_dataset import load_model, MultiChannelNormalizer, \
    FieldNormalizer, model_params_track_24, model_params_track_48, model_params_track_72, model_params_track_96

# 实例化训练模块
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    root_dir_field = '/data4/wxz_data/typhoon_intensity_bc/field_data_extraction'
    root_dir_value = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
    root_dir_target = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
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
    for forecast_time in [24, 48, 72, 96]:
        if forecast_time == 24:
            track_model = load_model('./data_file/BCNet/typhoon_track_24h/epoch=3611-step=7224.ckpt', device,
                                     model_params_track_24)
            field_normalizer = field_normalizer_24
            test_data_set = test_csv_file_24
        elif forecast_time == 48:
            track_model = load_model('./data_file/BCNet/typhoon_track_48h/epoch=2851-step=5704.ckpt', device,
                                     model_params_track_48)
            field_normalizer = field_normalizer_48
            test_data_set = test_csv_file_48
        elif forecast_time == 72:
            track_model = load_model('./data_file/BCNet/typhoon_track_72h/epoch=1891-step=3784.ckpt', device,
                                     model_params_track_72)
            field_normalizer = field_normalizer_72
            test_data_set = test_csv_file_72
        else:
            track_model = load_model('./data_file/BCNet/typhoon_track_96h/epoch=3031-step=3032.ckpt', device,
                                     model_params_track_96)
            field_normalizer = field_normalizer_96
            test_data_set = test_csv_file_96
        data_info = pd.read_csv(test_data_set)
        # 初始化列表存储数据
        latitude_true = []
        longitude_true = []
        latitude_origin = []
        longitude_origin = []
        latitude_corrected = []
        longitude_corrected = []
        origin_distance_arr = []
        forecast_distance_arr = []
        for index, row in data_info.iterrows():
            print(index)
            field_path = os.path.join(root_dir_field, f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}.pt")
            value_input_path = os.path.join(root_dir_value,
                                            f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_input.pt")
            target_path = os.path.join(root_dir_target,
                                       f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_target.pt")
            field_data = torch.load(field_path)
            value_input = torch.load(value_input_path)
            target = torch.load(target_path)
            field_input = torch.tensor(field_normalizer.transform(field_data)).unsqueeze(0)
            # track
            value_input = value_input[:, :2]
            target = target[:, :2]
            position = torch.tensor(position_normalizer.transform(value_input))
            position = torch.tensor(position).unsqueeze(0)
            # dot = make_dot(track_model(field_input, position), params=dict(track_model.named_parameters()))
            # dot.render('BCModel_graph', format='png')
            corrected_positions = track_model(field_input, position)[0]
            corrected_positions_real = position_normalizer.inverse_transform(corrected_positions.detach().cpu().numpy())
            value_input = value_input.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            corrected_positions_real[0] = target[0]
            # 真实位置、原始预测位置和订正后的预测位置
            latitude_true.extend(target[:, 1])
            longitude_true.extend(target[:, 0])
            latitude_origin.extend(value_input[:, 1])
            longitude_origin.extend(value_input[:, 0])
            latitude_corrected.extend(corrected_positions_real[:, 1])
            longitude_corrected.extend(corrected_positions_real[:, 0])
            forecast_distance = []
            origin_distance = []
            for i in range(0, len(target)):
                actual_position = (target[i, 1], target[i, 0])
                origin_position = (value_input[i, 1], value_input[i, 0])
                if i == 0:
                    forecast_position = (target[i, 1], target[i, 0])
                else:
                    forecast_position = (corrected_positions_real[i, 1], corrected_positions_real[i, 0])
                forecast_diff = haversine(forecast_position, actual_position)
                origin_diff = haversine(origin_position, actual_position)
                origin_distance.append(origin_diff)
                forecast_distance.append(forecast_diff)
            origin_distance_arr.append(origin_distance)
            forecast_distance_arr.append(forecast_distance)
        origin_distance_arr = np.array(origin_distance_arr)
        forecast_distance_arr = np.array(forecast_distance_arr)
        data_to_save = pd.DataFrame({
            'Latitude True': latitude_true,
            'Longitude True': longitude_true,
            'Latitude Origin': latitude_origin,
            'Longitude Origin': longitude_origin,
            'Latitude Corrected': latitude_corrected,
            'Longitude Corrected': longitude_corrected
        })
        path_save = f'./data_file/forecast_diff/typhoon_track_{forecast_time}h'
        np.save(f'{path_save}/origin_distance_arr.npy', origin_distance_arr)
        np.save(f'{path_save}/forecast_distance_arr.npy', forecast_distance_arr)
        position_data = data_to_save
        origin_distance_arr = origin_distance_arr.flatten()
        forecast_distance_arr = forecast_distance_arr.flatten()
        if len(position_data) == len(origin_distance_arr) and len(position_data) == len(forecast_distance_arr):
            # 添加新列
            position_data['Origin Distance'] = origin_distance_arr
            position_data['Corrected Distance'] = forecast_distance_arr
        else:
            print("Error: The length of the arrays does not match the number of rows in the DataFrame.")

        # 这里你可以保存更新后的 DataFrame 或进行其他处理
        # 例如，保存回 CSV
        position_data.to_csv(f'{path_save}/position_data.csv', index=False)
