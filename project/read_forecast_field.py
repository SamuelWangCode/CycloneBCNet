import os

import pandas as pd
import xarray as xr
import numpy as np


def find_nearest_index(array, value):
    """找到最接近指定值的索引"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def track_typhoon(df_real, typhoon_id, date, ds, initial_lon, initial_lat):
    ny, nx = len(ds['lat']), len(ds['lon'])
    track_info = {
        'lons': [],
        'lats': [],
        'pmin': [],
        'vmax': []
    }

    date_int = int(date)
    real_track = df_real[(df_real['ID'] == typhoon_id) & (df_real['Date'] >= date_int)]
    real_lons = real_track['Longitude'].tolist()
    real_lats = real_track['Latitude'].tolist()
    for t in range(len(ds['time'])):
        threshold = 4 if t <= 4 else 8 if t <= 8 else 14 if t <= 12 else 24
        msl = ds['msl'].isel(time=t)
        u10 = ds['10u'].isel(time=t)
        v10 = ds['10v'].isel(time=t)
        wspd10 = np.sqrt(u10 ** 2 + v10 ** 2)

        # 获取当前台风中心位置的索引
        lat_idx = np.abs(ds['lat'].values - initial_lat).argmin()
        lon_idx = np.abs(ds['lon'].values - (initial_lon % 360)).argmin()
        if t == 0:
            track_info['lats'].append(initial_lat)
            track_info['lons'].append(initial_lon)
        # 定义搜索半径
        search_radius = 4
        j_start = max(lat_idx - search_radius, 0)
        j_end = min(lat_idx + search_radius + 1, ny)
        i_start = max(lon_idx - search_radius, 0)
        i_end = min(lon_idx + search_radius + 1, nx)

        # 获取搜索区域内的海平面气压和风速
        sub_msl = msl.isel(lat=slice(j_start, j_end), lon=slice(i_start, i_end)) / 100
        sub_wspd10 = wspd10.isel(lat=slice(j_start, j_end), lon=slice(i_start, i_end))

        # 在搜索区域内找到最小气压和最大风速的位置
        min_pressure_idx = sub_msl.argmin().values.item()
        max_wind_speed_idx = sub_wspd10.argmax().values.item()

        # 更新台风中心位置
        min_lat_idx = j_start + min_pressure_idx // sub_msl.shape[1]
        min_lon_idx = i_start + min_pressure_idx % sub_msl.shape[1]
        initial_lat = ds['lat'][min_lat_idx].item()
        initial_lon = ds['lon'][min_lon_idx].item()
        if len(real_lons) > t:
            actual_lon, actual_lat = real_lons[t], real_lats[t]
            distance = max(abs(actual_lon - initial_lon), abs(actual_lat - initial_lat))
            if distance > threshold:
                print(f"Adjusting position at time step {t} due to excessive error")
                for j in range(1, t):
                    actual_lon, actual_lat = real_lons[j], real_lats[j]
                    lat_idx_actual = np.abs(ds['lat'].values - actual_lat).argmin()
                    lon_idx_actual = np.abs(ds['lon'].values - (actual_lon % 360)).argmin()
                    search_radius_actual = 4 if t <= 4 else 8 if t <= 8 else 14 if t <= 12 else 24
                    j_start = max(lat_idx_actual - search_radius_actual, 0)
                    j_end = min(lat_idx_actual + search_radius_actual, ny) + 1
                    i_start = max(lon_idx_actual - search_radius_actual, 0)
                    i_end = min(lon_idx_actual + search_radius_actual, nx) + 1

                    # 获取搜索区域内的海平面气压和风速
                    sub_msl = msl.isel(lat=slice(j_start, j_end), lon=slice(i_start, i_end)) / 100
                    sub_wspd10 = wspd10.isel(lat=slice(j_start, j_end), lon=slice(i_start, i_end))
                    # 在搜索区域内找到最小气压和最大风速的位置
                    min_pressure_idx = sub_msl.argmin().values.item()
                    max_wind_speed_idx = sub_wspd10.argmax().values.item()

                    # 更新台风中心位置
                    min_lat_idx = j_start + min_pressure_idx // sub_msl.shape[1]
                    min_lon_idx = i_start + min_pressure_idx % sub_msl.shape[1]
                    track_info['lats'][j] = ds['lat'][min_lat_idx].item()
                    track_info['lons'][j] = ds['lon'][min_lon_idx].item()
                    track_info['pmin'][j] = sub_msl.isel(lat=min_pressure_idx // sub_msl.shape[1],
                                                         lon=min_pressure_idx % sub_msl.shape[1]).values.item()
                    track_info['vmax'][j] = sub_wspd10.isel(lat=max_wind_speed_idx // sub_wspd10.shape[1],
                                                            lon=max_wind_speed_idx % sub_wspd10.shape[1]).values.item()
                    initial_lat = ds['lat'][min_lat_idx].item()
                    initial_lon = ds['lon'][min_lon_idx].item()
            if t == 0:
                track_info['pmin'].append(sub_msl.isel(lat=min_pressure_idx // sub_msl.shape[1],
                                                       lon=min_pressure_idx % sub_msl.shape[1]).values.item())
                track_info['vmax'].append(sub_wspd10.isel(lat=max_wind_speed_idx // sub_wspd10.shape[1],
                                                          lon=max_wind_speed_idx % sub_wspd10.shape[1]).values.item())
            else:
                track_info['lons'].append(initial_lon)
                track_info['lats'].append(initial_lat)
                track_info['pmin'].append(sub_msl.isel(lat=min_pressure_idx // sub_msl.shape[1],
                                                       lon=min_pressure_idx % sub_msl.shape[1]).values.item())
                track_info['vmax'].append(sub_wspd10.isel(lat=max_wind_speed_idx // sub_wspd10.shape[1],
                                                          lon=max_wind_speed_idx % sub_wspd10.shape[1]).values.item())
        else:
            print('real track over.')
            break

    return track_info


def process_all_forecasts(directory, typhoons, save_file):
    results = []
    # 获取所有.nc文件
    for file in os.listdir(directory):
        print(file)
        if file.startswith('prediction') and file.endswith('.nc'):
            date_str = file.split('_')[1]  # 获取日期
            forecast_time = date_str[:8] + date_str[9:11]  # YYYYMMDDHH
            typhoon_data = typhoons[typhoons['Date'].astype(str) == forecast_time]
            if not typhoon_data.empty:
                ds = xr.open_dataset(os.path.join(directory, file))
                for _, row in typhoon_data.iterrows():
                    track_info = track_typhoon(typhoons, row['ID'], forecast_time, ds, row['Longitude'], row['Latitude'])
                    results.append({
                        'typhoon_id': row['ID'],
                        'name': row['Name'],
                        'date': forecast_time,
                        'track_info': track_info
                    })
                ds.close()
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='typhoon_id')
    results_df.to_csv(save_file, index=False)


if __name__ == '__main__':
    # 加载台风数据
    typhoons = pd.read_csv('./data_file/typhoons.csv')
    # 示例使用
    directory = './outputs'
    process_all_forecasts(directory, typhoons, './data_file/typhoon_tracking_results.csv')
