from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json


def plot_typhoon_path(typhoon_id, date):
    # 读取台风追踪结果数据和真实台风路径数据
    df = pd.read_csv('./data_file/typhoon_tracking_results.csv')
    df_real = pd.read_csv('./data_file/typhoons.csv')

    # 选取指定台风ID和起报时间的记录
    record = df[(df['typhoon_id'] == typhoon_id) & (df['date'] == date)]
    real_track = df_real[(df_real['ID'] == typhoon_id) & (df_real['Date'] >= date)]

    if record.empty:
        print("没有找到指定的台风预报记录。")
        return
    if real_track.empty:
        print("没有找到指定的真实台风路径记录。")
        return

    # 解析track_info列的JSON数据
    track_info = json.loads(record.iloc[0]['track_info'].replace("'", "\""))

    # 获取经纬度数据
    lons = track_info['lons']
    lats = track_info['lats']
    real_lons = real_track['Longitude'].tolist()
    real_lats = real_track['Latitude'].tolist()

    # 创建绘图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(
        [min(lons + real_lons) - 5, max(lons + real_lons) + 5, min(lats + real_lats) - 5, max(lats + real_lats) + 5],
        crs=ccrs.PlateCarree())

    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # 绘制预报台风路径
    ax.plot(lons, lats, marker='o', color='b', linestyle='-', markersize=5, label='Forecast Track')
    ax.scatter(lons[0], lats[0], color='red', zorder=5, label='Start Position')  # 起始点

    # 绘制真实台风路径
    ax.plot(real_lons, real_lats, marker='o', color='g', linestyle='-', markersize=5, label='Actual Track')
    ax.scatter(real_lons[0], real_lats[0], color='orange', zorder=5, label='Real Start Position')  # 真实起始点

    # 添加网格线和标签
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 设置图例和标签
    plt.title(f'Typhoon {typhoon_id} ({date}) Track Comparison')
    plt.legend(loc='upper left')
    plt.savefig('track.png')


# 示例调用
plot_typhoon_path(2308, 2023081300)  # 调用时可以修改ID和日期来绘制其他台风路径
