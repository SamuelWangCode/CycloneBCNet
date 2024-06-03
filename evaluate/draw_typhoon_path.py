import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

if __name__ == '__main__':
    typhoon_name = 'KHANUN'
    # 加载数据
    df = pd.read_csv(f'./evaluate/case_study/position_data.csv')
    df_intensity = pd.read_csv(f'./evaluate/case_study/intensity_data.csv')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 24

    # 创建一个图形和三个子图
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 创建绘图
    # 添加起始点
    start_lon, start_lat = 132.8, 18.7
    start_row = {'Latitude True': start_lat, 'Longitude True': start_lon,
                 'Latitude Origin': start_lat, 'Longitude Origin': start_lon,
                 'Latitude Corrected': start_lat, 'Longitude Corrected': start_lon}
    # 在数据集的开始处插入起始点
    df = pd.concat([pd.DataFrame([start_row]), df], ignore_index=True)

    # 生成时间轴
    num_steps = len(df_intensity)
    time_hours = [i * 6 for i in range(num_steps)]  # 每个step代表6小时
    time_labels = [f"{hours}" for hours in time_hours]  # 创建小时标签

    display_step = 2  # 设置显示间隔
    displayed_ticks = range(0, num_steps, display_step)  # 获取显示的刻度
    displayed_labels = time_labels[::display_step]  # 获取显示的标签
    # 第一个子图 - 台风路径图
    ax1 = axes[0]
    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.set_extent([115, 140, 10, 35], crs=ccrs.PlateCarree())  # 设置地图范围
    ax1.add_feature(cfeature.COASTLINE)
    ax1.plot(df['Longitude True'], df['Latitude True'], 'r-o', label='CMA Best-track Path', transform=ccrs.Geodetic(),
             linewidth=4)
    ax1.plot(df['Longitude Origin'], df['Latitude Origin'], 'b--o', label='TianXing Path',
             transform=ccrs.Geodetic(), linewidth=4)
    ax1.plot(df['Longitude Corrected'], df['Latitude Corrected'], '-.o', color='orange', label='CycloneBCNet Path',
             transform=ccrs.Geodetic(), linewidth=4)
    # 第二个子图 - Vmax
    ax2 = axes[1]
    ax2.plot(df_intensity['Vmax True'], 'r-^', label='CMA Best-track Vmax', linewidth=4)
    ax2.plot(df_intensity['Vmax Origin'], 'b--^', label='TianXing Vmax', linewidth=4)
    ax2.plot(df_intensity['Vmax Corrected'], '-.^', color='orange', label='CycloneBCNet Vmax', linewidth=4)
    ax2.set_xlabel('Forecast Duration (hour)')
    ax2.set_ylabel('Wind Speed (m s$^{-1}$)')
    ax2.set_title('Maximum Wind Speed Over Time')
    ax2.set_xticks(displayed_ticks)
    ax2.set_xticklabels(displayed_labels)
    ax2.set_box_aspect(1)  # 保持宽高比

    # 第三个子图 - Pmin
    ax3 = axes[2]
    ax3.plot(df_intensity['Pmin True'], 'r-o', label='CMA Best-track Pmin', linewidth=4)
    ax3.plot(df_intensity['Pmin Origin'], 'b--o', label='TianXing Pmin', linewidth=4)
    ax3.plot(df_intensity['Pmin Corrected'], '-.o', color='orange', label='CycloneBCNet Pmin', linewidth=4)
    ax3.set_xlabel('Forecast Duration (hour)')
    ax3.set_ylabel('Pressure (hPa)')
    ax3.set_title('Minimum Pressure Over Time')
    ax3.set_xticks(displayed_ticks)
    ax3.set_xticklabels(displayed_labels)
    ax3.set_box_aspect(1)  # 保持宽高比

    # 调整子图间距
    plt.subplots_adjust(wspace=0.3)
    # 显示图
    plt.savefig(f'./evaluate/case_study/typhoon_combined_plot.png', dpi=600, bbox_inches='tight')
