import json

import pandas as pd
import torch

# 加载数据
forecast_df = pd.read_csv('./data_file/forecast_instances.csv')
tracking_df = pd.read_csv('./data_file/typhoon_tracking_results.csv')
typhoons_df = pd.read_csv('./data_file/typhoons.csv')

# 转换列名以确保匹配
tracking_df.rename(columns={'typhoon_id': 'ID', 'date': 'Start Date'}, inplace=True)
typhoons_df.rename(columns={'Date': 'Start Date'}, inplace=True)
# 合并数据
merged_df = forecast_df.merge(tracking_df, on=['ID', 'Start Date'])
final_df = merged_df.merge(typhoons_df, on=['ID', 'Start Date'])

# 数据提取和格式化
for index, row in final_df.iterrows():
    print(index)
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
            (typhoons_df['ID'] == row['ID']) & (typhoons_df['Start Date'] == int(forecast_time.strftime('%Y%m%d%H')))]
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
               f'/data4/wxz_data/typhoon_intensity_bc/value_data_extraction/{row["ID"]}_{row["Start Date"]}_{row["Forecast Hour"]}_input.pt')
    torch.save(target_tensor,
               f'/data4/wxz_data/typhoon_intensity_bc/value_data_extraction/{row["ID"]}_{row["Start Date"]}_{row["Forecast Hour"]}_target.pt')
