import math
import os
import torch
import pandas as pd
from torch import tensor

from typhoon_intensity_bc.project.construct_dataset import load_model, MultiChannelNormalizer, FieldNormalizer, \
    model_params_track_96


def load_and_predict(field_data_path, value_input_path, model, field_normalizer, position_normalizer):
    # Load the field data
    field_data = torch.load(field_data_path)
    value_input = torch.load(value_input_path)
    field_input = tensor(field_normalizer.transform(field_data)).unsqueeze(0)

    # Assuming `value_input` is part of the loaded data, otherwise you need to modify how it's obtained
    value_input = value_input[:, :2]
    position = tensor(position_normalizer.transform(value_input))
    position = position.unsqueeze(0)

    # Perform the prediction
    corrected_positions = model(field_input, position)[0]
    corrected_positions_real = position_normalizer.inverse_transform(corrected_positions.detach().cpu().numpy())
    corrected_positions_real[0] = value_input[0, :2]
    return corrected_positions_real


def save_predicted_coordinates(predicted_coords, typhoon_id, start_date, forecast_hour):
    save_path = f'/data4/wxz_data/typhoon_intensity_bc/track_forecast_data/{typhoon_id}_{start_date}_{forecast_hour}.pt'
    torch.save(predicted_coords, save_path)
    print(f'Saved predicted coordinates to {save_path}')


def main():
    forecast_instances_df = pd.read_csv('./data_file/forecast_instances.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    track_model = load_model('./data_file/BCNet/typhoon_track_96h/epoch=3031-step=3032.ckpt', device,
                             model_params_track_96)
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

    for index, row in forecast_instances_df.iterrows():
        if row['Forecast Hour'] == 96:
            print(index)
            typhoon_id = row['ID']
            start_date = row['Start Date']
            forecast_hour = row['Forecast Hour']

            # Construct the path to the field data file
            field_data_path = f'/data4/wxz_data/typhoon_intensity_bc/field_data_extraction/{typhoon_id}_{start_date}_{forecast_hour}.pt'
            value_input_path = f'/data4/wxz_data/typhoon_intensity_bc/value_data_extraction/{typhoon_id}_{start_date}_{forecast_hour}_input.pt'
            if os.path.exists(field_data_path):
                predicted_coords = load_and_predict(field_data_path, value_input_path, track_model, field_normalizer,
                                                    position_normalizer)
                save_predicted_coordinates(predicted_coords, typhoon_id, start_date, forecast_hour)


if __name__ == '__main__':
    main()
