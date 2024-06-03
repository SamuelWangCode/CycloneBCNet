import os
import math
import pandas as pd
import numpy as np
from haversine import haversine
import torch

from typhoon_intensity_bc.project.construct_dataset import load_model, MultiChannelNormalizer, \
    FieldNormalizer, model_params_track_96, model_params_96

if __name__ == '__main__':
    typhoon_name = 'KHANUN'
    # 实例化训练模块
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    track_model = load_model('./data_file/BCNet/typhoon_track_96h/epoch=3031-step=3032.ckpt', device,
                             model_params_track_96)
    intensity_model = load_model('./data_file/BCNet/typhoon_intensity_96h/epoch=2034-step=2035.ckpt', device,
                             model_params_96)
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
    field_normalizer = field_normalizer_96
    root_dir_field = './evaluate/case_study/'
    root_dir_value = './evaluate/case_study/'
    root_dir_target = './evaluate/case_study/'
    root_dir_track = './evaluate/case_study/'
    test_data_set = './evaluate/case_study/forecast_instances.csv'
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
        # 真实位置、原始预测位置和订正后的预测位置
        latitude_true.extend(target[1:, 1])
        longitude_true.extend(target[1:, 0])
        latitude_origin.extend(value_input[1:, 1])
        longitude_origin.extend(value_input[1:, 0])
        latitude_corrected.extend(corrected_positions_real[1:, 1])
        longitude_corrected.extend(corrected_positions_real[1:, 0])
        forecast_distance = []
        origin_distance = []
        for i in range(1, len(target)):
            actual_position = (target[i, 1], target[i, 0])
            origin_position = (value_input[i, 1], value_input[i, 0])
            forecast_position = (corrected_positions_real[i, 1], corrected_positions_real[i, 0])
            forecast_diff = haversine(forecast_position, actual_position)
            origin_diff = haversine(origin_position, actual_position)
            origin_distance.append(origin_diff)
            forecast_distance.append(forecast_diff)
    origin_distance_arr = np.array(origin_distance)
    forecast_distance_arr = np.array(forecast_distance)
    data_to_save = pd.DataFrame({
        'Latitude True': latitude_true,
        'Longitude True': longitude_true,
        'Latitude Origin': latitude_origin,
        'Longitude Origin': longitude_origin,
        'Latitude Corrected': latitude_corrected,
        'Longitude Corrected': longitude_corrected
    })
    corrected_path = torch.tensor(corrected_positions_real[1:])
    corrected_path = torch.cat((torch.load(target_path)[0, :2].unsqueeze(0), corrected_path), dim=0)
    torch.save(corrected_path, f'./evaluate/case_study/forecast_track_{typhoon_name}.pt')
    position_data = data_to_save
    if len(position_data) == len(origin_distance_arr) and len(position_data) == len(forecast_distance_arr):
        # 添加新列
        position_data['Origin Distance'] = origin_distance_arr
        position_data['Corrected Distance'] = forecast_distance_arr
    else:
        print("Error: The length of the arrays does not match the number of rows in the DataFrame.")

    # 这里你可以保存更新后的 DataFrame 或进行其他处理
    # 例如，保存回 CSV
    position_data.to_csv(f'./evaluate/case_study/position_data_{typhoon_name}.csv', index=False)
    # 初始化列表存储数据
    vmax_true = []
    pmin_true = []
    vmax_origin = []
    pmin_origin = []
    vmax_corrected = []
    pmin_corrected = []
    for index, row in data_info.iterrows():
        print(index)
        field_path = os.path.join(root_dir_field, f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}.pt")
        value_input_path = os.path.join(root_dir_value,
                                        f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_input.pt")
        track_input_path = os.path.join(root_dir_track,
                                        f"forecast_track_{typhoon_name}.pt")
        target_path = os.path.join(root_dir_target,
                                   f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_target.pt")
        field_data = torch.load(field_path)
        value_input = torch.load(value_input_path)
        track_input = torch.load(track_input_path)
        field_input = torch.tensor(field_normalizer.transform(field_data)).unsqueeze(0)
        # track
        intensity_input = value_input[:, 2:]
        copy_intensity_input = value_input[:, 2:].detach().cpu().numpy()
        vmax_origin.extend(copy_intensity_input[:, 1])
        pmin_origin.extend(copy_intensity_input[:, 0])
        intensity_input = torch.tensor(intensity_normalizer.transform(intensity_input))
        track_input = torch.tensor(position_normalizer.transform(track_input))
        value_input = torch.cat((track_input, intensity_input), dim=1)
        target = torch.load(target_path)[:, 2:]
        position = torch.tensor(value_input).unsqueeze(0)
        corrected_intensity = intensity_model(field_input, position)[0]
        corrected_intensity_real = intensity_normalizer.inverse_transform(corrected_intensity.detach().cpu().numpy())
        value_input = value_input[:, 2:].detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        vmax_true.extend(target[:, 1])
        pmin_true.extend(target[:, 0])
        vmax_corrected.extend(corrected_intensity_real[:, 1])
        pmin_corrected.extend(corrected_intensity_real[:, 0])
    data_to_save = pd.DataFrame({
        'Vmax True': vmax_true,
        'Pmin True': pmin_true,
        'Vmax Origin': vmax_origin,
        'Pmin Origin': pmin_origin,
        'Vmax Corrected': vmax_corrected,
        'Pmin Corrected': pmin_corrected
    })
    data_to_save.to_csv(f'./evaluate/case_study/intensity_data_{typhoon_name}.csv', index=False)
