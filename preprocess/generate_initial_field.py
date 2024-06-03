import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import os
import shutil


# 定义辅助函数
def check_and_copy(source, destination):
    if not os.path.exists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)
    else:
        print(destination + 'exists.')


def prepare_initial_conditions(row, forecast_hours):
    date_str = str(row['Date'])
    print('prepare input field for ' + date_str)
    base_time = datetime.strptime(date_str, "%Y%m%d%H")
    date_str = base_time.strftime("%Y%m%dT%H")
    pl_file_src = f'../inputs/{base_time.strftime("%Y%m")}_pl.nc'
    sfc_file_src = f'../inputs/{base_time.strftime("%Y%m")}_sfc.nc'
    output_dir = f'../inputs/{date_str}/'
    os.makedirs(output_dir, exist_ok=True)

    # Load the data for the specific time slice
    with xr.open_dataset(pl_file_src) as ds_pl:
        ds_pl = ds_pl.sel(time=base_time)
        ds_pl.to_netcdf(f'{output_dir}pl.nc')

    with xr.open_dataset(sfc_file_src) as ds_sfc:
        ds_sfc = ds_sfc.sel(time=base_time)
        ds_sfc.to_netcdf(f'{output_dir}sfc.nc')


def generate_forecast_instances(df):
    forecast_hours = [24, 48, 72, 96]
    result = []
    for tid, group in df.groupby('ID'):
        for index, row in group.iterrows():
            # 确保日期字段是字符串格式
            date_str = str(row['Date'])
            base_time = datetime.strptime(date_str, "%Y%m%d%H")
            for fh in forecast_hours:
                forecast_time = base_time + timedelta(hours=fh)
                if forecast_time <= datetime.strptime(str(group['Date'].max()), "%Y%m%d%H"):
                    max_category_during_forecast = group[(group['Date'] >= int(date_str)) &
                                                         (group['Date'] <= int(forecast_time.strftime("%Y%m%d%H")))]['Category'].max()
                    class_label = classify_typhoon(max_category_during_forecast)
                    prepare_initial_conditions(row, [fh])
                    result.append({
                        'ID': row['ID'],
                        'Name': row['Name'],
                        'Start Date': date_str,
                        'Forecast Hour': fh,
                        'Class': class_label,
                    })
    return pd.DataFrame(result)


def classify_typhoon(max_category):
    if max_category >= 6:
        return 'Super TY'
    elif max_category == 5:
        return 'STY'
    elif max_category == 4:
        return 'TY'
    elif max_category == 3:
        return 'STS'
    else:
        return 'TS'


def save_forecast_instance(row):
    forecast_hours = [24, 48, 72, 96]
    prepare_initial_conditions(row, forecast_hours)


# 读取 CSV 文件
df = pd.read_csv("./data_file/typhoons.csv")
forecast_df = generate_forecast_instances(df)
forecast_df.to_csv('./data_file/forecast_instances.csv', index=False)
