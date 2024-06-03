from typing import Dict

import numpy as np
import pandas as pd
import torch
import xarray as xr
import os
import json


def extract_data(prediction_file, track_info, grid_expansion, forecast_hour):
    ds = xr.open_dataset(prediction_file)
    # 假设数据集中包含有'longitude'和'latitude'变量
    lons = ds['lon']
    lats = ds['lat']

    extracted_data = []
    T = forecast_hour // 6 + 1
    C = 73
    W = grid_expansion[forecast_hour] * 2 + 1
    H = grid_expansion[forecast_hour] * 2 + 1
    tensor_data = np.empty((T, C, W, H))
    for t in range(T):
        lon = track_info['lons'][t]
        lat = track_info['lats'][t]
        lon_idx = int(abs(lons - lon).argmin())
        lat_idx = int(abs(lats - lat).argmin())

        # 根据预报时长确定扩展的格点范围
        expansion = grid_expansion[forecast_hour]
        lon_slice = slice(max(0, lon_idx - expansion), min(len(lons), lon_idx + expansion + 1))
        lat_slice = slice(max(0, lat_idx - expansion), min(len(lats), lat_idx + expansion + 1))
        data_tensor = ds.isel(lon=lon_slice, lat=lat_slice, time=t)
        data_vars = list(data_tensor.data_vars)
        stacked_data = np.stack([data_tensor[var].values for var in data_vars], axis=0)
        tensor_data[t] = stacked_data
        # 提取数据
        extracted_data.append(data_tensor)
    return xr.concat(extracted_data, dim='time'), tensor_data


def main():
    forecast_instances_df = pd.read_csv('./data_file/forecast_instances.csv')
    tracking_df = pd.read_csv('./data_file/typhoon_tracking_results.csv')
    grid_expansions: dict[int, int] = {24: 26, 48: 28, 72: 32, 96: 36}

    for index, row in forecast_instances_df.iterrows():
        typhoon_id = row['ID']
        start_date = row['Start Date']
        forecast_hour = row['Forecast Hour']

        # 构造预报场文件路径
        prediction_filename = f'./outputs/prediction_{str(start_date)[:-2]}T{str(start_date)[-2:]}.nc'
        record = tracking_df[(tracking_df['typhoon_id'] == typhoon_id) & (tracking_df['date'] == start_date)]
        track_info_json = record.iloc[0]['track_info']
        track_info_json = track_info_json.replace("'", '"')  # 将单引号替换为双引号
        track_info = json.loads(track_info_json)

        # 提取数据
        if os.path.exists(prediction_filename):
            data, tensor_data = extract_data(prediction_filename, track_info, grid_expansions, forecast_hour)
            # 保存数据
            save_path = f'/data4/wxz_data/typhoon_intensity_bc/field_data_extraction/{typhoon_id}_{start_date}_{forecast_hour}.nc'
            tensor_save_path = f'/data4/wxz_data/typhoon_intensity_bc/field_data_extraction/{typhoon_id}_{start_date}_{forecast_hour}.pt'
            tensor = torch.from_numpy(tensor_data).float()
            data.to_netcdf(save_path)
            torch.save(tensor, tensor_save_path)
            print(f'Saved extracted data to {save_path}')


if __name__ == '__main__':
    main()
