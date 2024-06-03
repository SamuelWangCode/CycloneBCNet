import csv
import datetime
import gc
import json
import os

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
from model import weaformerV2, ordering
from typhoon_intensity_bc.preprocess.get_forecast_field import load_model_and_initial_field
from typhoon_intensity_bc.preprocess.read_CMA_to_CSV import Typhoon, parse_path_record
from typhoon_intensity_bc.project.extract_data_from_typhoon import extract_data
from typhoon_intensity_bc.project.read_forecast_field import process_all_forecasts


def check_and_download_era5(year, month, day, hour):
    if os.path.exists(f'/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/inputs/{year}{month}{day}T{hour}'):
        print('era5 file exists.')
    else:
        print('开始下载era5文件')
        os.makedirs(f'/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/inputs/{year}{month}{day}T{hour}')
        c = cdsapi.Client()

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "surface_pressure",
                    "mean_sea_level_pressure",
                    "total_column_water_vapour",
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "sea_surface_temperature",
                ],
                'year': year,
                'month': month,
                'day': day,
                'time': f'{hour}:00',
                'format': 'netcdf',
            },
            f'/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/inputs/{year}{month}{day}T{hour}/sfc.nc')

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "geopotential",
                    "relative_humidity",
                ],
                'pressure_level': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
                'year': year,
                'month': month,
                'day': day,
                'time': f'{hour}:00',
                'format': 'netcdf',
            },
            f'/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/inputs/{year}{month}{day}T{hour}/pl.nc')


def parse_file(filename, typhoon_id):
    current_typhoon = None
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('66666'):
                parts = line.split()
                id = parts[4]
                if id == typhoon_id:  # Skip typhoon with ID '0000'
                    name = parts[7].strip()
                    current_typhoon = Typhoon(id, name)
                    typhoon = current_typhoon
                else:
                    current_typhoon = None
            else:
                if current_typhoon is not None:
                    record = parse_path_record(line)
                    if record is not None:  # 检查 record 是否有效
                        # Check if record is within the specified geographic boundaries
                        if 0 <= record['lat'] <= 70 and 100 <= record['lon'] <= 180:
                            current_typhoon.add_record(record)
    return typhoon


def save_to_csv(typhoon, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Assuming you want to write headers
        writer.writerow(['ID', 'Name', 'Date', 'Category', 'Latitude', 'Longitude', 'Pressure', 'Wind Speed'])
        for record in typhoon.records:
            writer.writerow(
                [typhoon.id, typhoon.name, record['date'], record['category'], record['lat'], record['lon'],
                 record['pressure'], record['wind_speed']])


def predict(model_wrapper, input_path):
    print('Start predict')
    forecast_model, initial_field = load_model_and_initial_field(
        model_wrapper,
        input_path=input_path)
    outputs = [initial_field]
    for i in range(int(lead_time / hour_step)):
        output = checkpoint(forecast_model, outputs[i])
        outputs.append(output)
        del output  # 显式删除不再需要的 Tensor
        gc.collect()
        torch.cuda.empty_cache()  # 清空未使用的缓存
    for i in range(len(outputs)):
        outputs[i] = model_wrapper.normalise(outputs[i], reverse=True)
    return outputs


if __name__ == '__main__':
    start_date = 2023073000
    year = '2023'
    month = '07'
    day = '30'
    hour = '00'
    typhoon_id = '2306'
    typhoon_name = 'KHANUN'
    check_and_download_era5(year, month, day, hour)
    print('创建typhoons.csv')
    file = f'./data_file/CMA_typhoon/CH{year}BST.txt'
    typhoon = parse_file(file, typhoon_id)
    save_to_csv(typhoon, './evaluate/case_study/typhoons.csv')
    data = {
        'ID': [typhoon_id],
        'Name': [typhoon_name],
        'Start Date': [start_date],
        'Forecast Hour': [96],
        'Class': ['Super TY']
    }

    # 将字典转换为pandas DataFrame
    df = pd.DataFrame(data)

    # 将DataFrame写入CSV文件
    csv_file = './evaluate/case_study/forecast_instances.csv'  # 指定文件名
    df.to_csv(csv_file, index=False)  # 写入文件，不包含行索引

    print(f'CSV file "{csv_file}" has been created.')  # 打印文件创建成功的消息
    print('开始预报')
    root_path = "/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/"
    hour_step = 6
    lead_time = 96
    model_wrapper = weaformerV2(root_path=root_path, lead_time=lead_time, hour_step=hour_step)
    input_path = f"../inputs/{str(start_date)[:8]}T{str(start_date)[-2:]}"
    start_date = datetime.datetime.strptime(str(start_date), "%Y%m%d%H")
    pl_data = xr.open_dataset(f"{input_path}/pl.nc")
    outputs = predict(model_wrapper, input_path)
    output_np = torch.stack(outputs).cpu().numpy().squeeze()
    # 时间步定义
    time_steps = pd.date_range(start=start_date, periods=output_np.shape[0], freq='6H')

    # 创建 Dataset 而不是 DataArray
    coords = {
        "time": time_steps,
        "lat": np.linspace(90, -90, 721),
        "lon": np.linspace(0, 359.75, 1440)
    }

    data_vars = {}
    for i, var in enumerate(ordering):
        data_vars[var] = (("time", "lat", "lon"), output_np[:, i, :, :])

    dataset = xr.Dataset(data_vars, coords=coords)

    # 文件保存路径
    output_filename = f'./evaluate/case_study/prediction_{start_date.strftime("%Y%m%dT%H")}.nc'
    dataset.to_netcdf(output_filename)
    print(f"Saved prediction to {output_filename}")
    print('读取预报场台风路径强度')
    # 加载台风数据
    typhoons = pd.read_csv('./evaluate/case_study/typhoons.csv')
    # 示例使用
    directory = './evaluate/case_study'
    process_all_forecasts(directory, typhoons, './evaluate/case_study/typhoon_tracking_results.csv')
    print('获取预报场和值输入张量')
    forecast_instances_df = pd.read_csv('./evaluate/case_study/forecast_instances.csv')
    tracking_df = pd.read_csv('./evaluate/case_study/typhoon_tracking_results.csv')
    grid_expansions: dict[int, int] = {24: 26, 48: 28, 72: 32, 96: 36}
    for index, row in forecast_instances_df.iterrows():
        typhoon_id = row['ID']
        start_date = row['Start Date']
        forecast_hour = row['Forecast Hour']

        # 构造预报场文件路径
        prediction_filename = f'./evaluate/case_study/prediction_{str(start_date)[:-2]}T{str(start_date)[-2:]}.nc'
        record = tracking_df[(tracking_df['typhoon_id'] == typhoon_id) & (tracking_df['date'] == start_date)]
        track_info_json = record.iloc[0]['track_info']
        track_info_json = track_info_json.replace("'", '"')  # 将单引号替换为双引号
        track_info = json.loads(track_info_json)

        # 提取数据
        if os.path.exists(prediction_filename):
            data, tensor_data = extract_data(prediction_filename, track_info, grid_expansions, forecast_hour)
            # 保存数据
            save_path = f'./evaluate/case_study/{typhoon_id}_{start_date}_{forecast_hour}.nc'
            tensor_save_path = f'./evaluate/case_study/{typhoon_id}_{start_date}_{forecast_hour}.pt'
            tensor = torch.from_numpy(tensor_data).float()
            data.to_netcdf(save_path)
            torch.save(tensor, tensor_save_path)
            print(f'Saved extracted data to {save_path}')
    # 加载数据
    forecast_df = pd.read_csv('./evaluate/case_study/forecast_instances.csv')
    tracking_df = pd.read_csv('./evaluate/case_study/typhoon_tracking_results.csv')
    typhoons_df = pd.read_csv('./evaluate/case_study/typhoons.csv')

    # 转换列名以确保匹配
    tracking_df.rename(columns={'typhoon_id': 'ID', 'date': 'Start Date'}, inplace=True)
    typhoons_df.rename(columns={'Date': 'Start Date'}, inplace=True)
    # 合并数据
    merged_df = forecast_df.merge(tracking_df, on=['ID', 'Start Date'])
    final_df = merged_df.merge(typhoons_df, on=['ID', 'Start Date'])

    # 数据提取和格式化
    for index, row in final_df.iterrows():
        start_date = pd.to_datetime(row['Start Date'], format='%Y%m%d%H')
        # 获取每个预报时长的时间步
        T = row['Forecast Hour'] // 6 + 1
        track_info = row['track_info']
        track_info = json.loads(track_info.replace("'", '"'))

        # 初始化tensor
        input_tensor = torch.zeros(T, 4)
        target_tensor = torch.zeros(T, 4)

        # 填充tensor
        for t in range(T):
            forecast_time = start_date + pd.Timedelta(hours=t * 6)
            matching_row = typhoons_df[
                (typhoons_df['ID'] == row['ID']) & (
                        typhoons_df['Start Date'] == int(forecast_time.strftime('%Y%m%d%H')))]
            input_tensor[t] = torch.tensor(
                [track_info['lons'][t], track_info['lats'][t], track_info['pmin'][t], track_info['vmax'][t]])
            if track_info['lons'][t] > 180:
                print(track_info['lons'][t])
                print('大')
            if track_info['lons'][t] < 0:
                print(track_info['lons'][t])
                print('小')
            if not matching_row.empty:
                target_tensor[t] = torch.tensor([
                    matching_row.iloc[0]['Longitude'],
                    matching_row.iloc[0]['Latitude'],
                    matching_row.iloc[0]['Pressure'],
                    matching_row.iloc[0]['Wind Speed']
                ])

        # 保存tensor
        torch.save(input_tensor,
                   f'./evaluate/case_study/{row["ID"]}_{row["Start Date"]}_{row["Forecast Hour"]}_input.pt')
        torch.save(target_tensor,
                   f'./evaluate/case_study/{row["ID"]}_{row["Start Date"]}_{row["Forecast Hour"]}_target.pt')
    print('finished')
