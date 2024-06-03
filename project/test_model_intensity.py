import os
import pandas as pd
import torch

from typhoon_intensity_bc.project.construct_dataset import load_model, MultiChannelNormalizer, \
    FieldNormalizer, model_params_24, model_params_48, model_params_72, model_params_96

if __name__ == '__main__':
    # 实例化训练模块
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
    root_dir_track = '/data4/wxz_data/typhoon_intensity_bc/track_forecast_data'
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
            intensity_model = load_model('./data_file/BCNet/typhoon_intensity_24h/epoch=1030-step=2062.ckpt', device,
                                         model_params_24)
            field_normalizer = field_normalizer_24
            test_data_set = test_csv_file_24
        elif forecast_time == 48:
            intensity_model = load_model('./data_file/BCNet/typhoon_intensity_48h/epoch=951-step=1904.ckpt', device,
                                         model_params_48)
            field_normalizer = field_normalizer_48
            test_data_set = test_csv_file_48
        elif forecast_time == 72:
            intensity_model = load_model('./data_file/BCNet/typhoon_intensity_72h/epoch=1372-step=2746.ckpt', device,
                                         model_params_72)
            field_normalizer = field_normalizer_72
            test_data_set = test_csv_file_72
        else:
            intensity_model = load_model('./data_file/BCNet/typhoon_intensity_96h/epoch=2034-step=2035.ckpt', device,
                                         model_params_96)
            field_normalizer = field_normalizer_96
            test_data_set = test_csv_file_96
        data_info = pd.read_csv(test_data_set)
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
                                            f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}.pt")
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
            corrected_intensity_real = intensity_normalizer.inverse_transform(
                corrected_intensity.detach().cpu().numpy())
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
        data_to_save.to_csv(f'./data_file/forecast_diff/typhoon_intensity_{forecast_time}h/intensity_data.csv',
                            index=False)
