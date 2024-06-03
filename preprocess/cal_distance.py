import pandas as pd
import numpy as np
import os
import json
from haversine import haversine


def convert_json(obj):
    # 将单引号转换为双引号
    obj = obj.replace("'", '"')
    try:
        return json.loads(obj)
    except json.decoder.JSONDecodeError:
        return None


def adjust_longitude(lon):
    if lon > 180:
        lon = lon - 360  # 如果经度超过180度，调整到-180到180的范围内
    elif lon < -180:
        lon = lon + 360  # 如果经度小于-180度，调整到-180到180的范围内
    return lon


def load_and_prepare_data():
    # 加载数据
    typhoons = pd.read_csv('./data_file/typhoons.csv')
    forecasts = pd.read_csv('./data_file/forecast_instances.csv')
    tracking_results = pd.read_csv('./data_file/typhoon_tracking_results.csv', converters={'track_info': convert_json})

    # 预处理，将时间字符串转换为可比较的格式
    typhoons['Date'] = pd.to_datetime(typhoons['Date'], format='%Y%m%d%H')
    forecasts['Start Date'] = pd.to_datetime(forecasts['Start Date'], format='%Y%m%d%H')
    tracking_results['date'] = pd.to_datetime(tracking_results['date'], format='%Y%m%d%H')

    return typhoons, forecasts, tracking_results


def calculate_errors(forecasts, typhoons, tracking_results):
    error_results = []
    # 处理每一个预报时长
    for hours in [24, 48, 72, 96]:
        print('forecast hours: ' + str(hours))
        specific_forecasts = forecasts[forecasts['Forecast Hour'] == hours]
        for index, forecast in specific_forecasts.iterrows():
            print(index)
            # 找到对应的真实路径数据和预报结果
            actual = typhoons[(typhoons['ID'] == forecast['ID']) & (typhoons['Date'] >= forecast['Start Date']) & (
                    typhoons['Date'] < forecast['Start Date'] + pd.Timedelta(hours=hours))]
            forecast_result = tracking_results[(tracking_results['typhoon_id'] == forecast['ID']) & (
                    tracking_results['date'] == forecast['Start Date'])]
            if forecast_result.empty or actual.empty:
                continue

            forecast_track_info = forecast_result.iloc[0]['track_info']
            forecast_lons, forecast_lats, forecast_pmins, forecast_vmaxs = forecast_track_info['lons'], \
                forecast_track_info['lats'], forecast_track_info['pmin'], forecast_track_info['vmax']

            # 计算误差
            max_distance_error = 0
            total_distance_error = 0
            max_pressure_error = 0
            total_pressure_error = 0
            max_wind_speed_error = 0
            total_wind_speed_error = 0
            count = 0

            for i in range(min(len(forecast_lons), len(actual))):
                if i == 0:
                    pressure_error = abs(forecast_pmins[i] - actual.iloc[i]['Pressure'])
                    wind_speed_error = abs(forecast_vmaxs[i] - actual.iloc[i]['Wind Speed'])

                    max_pressure_error = max(max_pressure_error, pressure_error)
                    max_wind_speed_error = max(max_wind_speed_error, wind_speed_error)

                    total_pressure_error += pressure_error
                    total_wind_speed_error += wind_speed_error
                else:
                    forecast_position = (forecast_lats[i], adjust_longitude(forecast_lons[i]))
                    actual_position = (actual.iloc[i]['Latitude'], adjust_longitude(actual.iloc[i]['Longitude']))
                    distance_error = haversine(forecast_position, actual_position)

                    pressure_error = abs(forecast_pmins[i] - actual.iloc[i]['Pressure'])
                    wind_speed_error = abs(forecast_vmaxs[i] - actual.iloc[i]['Wind Speed'])

                    max_distance_error = max(max_distance_error, distance_error)
                    max_pressure_error = max(max_pressure_error, pressure_error)
                    max_wind_speed_error = max(max_wind_speed_error, wind_speed_error)

                    total_distance_error += distance_error
                    total_pressure_error += pressure_error
                    total_wind_speed_error += wind_speed_error
                count += 1

            # 记录结果
            if count > 0:
                error_results.append({
                    'Typhoon ID': forecast['ID'],
                    'Forecast Start Date': forecast['Start Date'],
                    'Forecast Hour': hours,
                    'Max Distance Error (km)': max_distance_error,
                    'Average Distance Error (km)': total_distance_error / count,
                    'Max Pressure Error (hPa)': max_pressure_error,
                    'Average Pressure Error (hPa)': total_pressure_error / count,
                    'Max Wind Speed Error (m/s)': max_wind_speed_error,
                    'Average Wind Speed Error (m/s)': total_wind_speed_error / count
                })

    return pd.DataFrame(error_results)


# 主逻辑
typhoons, forecasts, tracking_results = load_and_prepare_data()
error_df = calculate_errors(forecasts, typhoons, tracking_results)
error_df.to_csv('./data_file/error_analysis.csv', index=False)

# 预报时长列表
forecast_hours = [24, 48, 72, 96]

summary_list = []

for hours in forecast_hours:
    specific_errors = error_df[error_df['Forecast Hour'] == hours]

    # 计算最大和平均误差
    max_distance_error = specific_errors['Max Distance Error (km)'].max()
    avg_distance_error = specific_errors['Average Distance Error (km)'].mean()
    max_pressure_error = specific_errors['Max Pressure Error (hPa)'].max()
    avg_pressure_error = specific_errors['Average Pressure Error (hPa)'].mean()
    max_wind_speed_error = specific_errors['Max Wind Speed Error (m/s)'].max()
    avg_wind_speed_error = specific_errors['Average Wind Speed Error (m/s)'].mean()

    # 构建结果字典并添加到列表中
    summary_list.append({
        'Forecast Hour': hours,
        'Max Distance Error (km)': max_distance_error,
        'Average Distance Error (km)': avg_distance_error,
        'Max Pressure Error (hPa)': max_pressure_error,
        'Average Pressure Error (hPa)': avg_pressure_error,
        'Max Wind Speed Error (m/s)': max_wind_speed_error,
        'Average Wind Speed Error (m/s)': avg_wind_speed_error
    })

    # 使用pd.concat来创建一个新的DataFrame
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv('./data_file/summary_error.csv')
