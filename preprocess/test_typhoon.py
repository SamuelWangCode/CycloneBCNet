import pandas as pd

# 加载数据
typhoons = pd.read_csv('./data_file/typhoons.csv')

# 确保数据按照台风ID和日期排序
typhoons.sort_values(by=['ID', 'Date'], inplace=True)

# 初始化变量来存储最大的纬度和经度变化
max_lat_change_below_30 = 0
max_lon_change_below_30 = 0
max_lat_change_above_30 = 0
max_lon_change_above_30 = 0

# 遍历每一个台风的记录
for _, group in typhoons.groupby('ID'):
    # 计算经纬度的变化
    lat_changes = group['Latitude'].diff().abs()
    lon_changes = group['Longitude'].diff().abs()
    lats = group['Latitude']

    # 检查变化并更新最大变化值
    for i in range(1, len(group)):
        if lats.iloc[i-1] < 30:
            if lat_changes.iloc[i] > max_lat_change_below_30:
                max_lat_change_below_30 = lat_changes.iloc[i]
            if lon_changes.iloc[i] > max_lon_change_below_30:
                max_lon_change_below_30 = lon_changes.iloc[i]
        else:
            if lat_changes.iloc[i] > max_lat_change_above_30:
                max_lat_change_above_30 = lat_changes.iloc[i]
            if lon_changes.iloc[i] > max_lon_change_above_30:
                max_lon_change_above_30 = lon_changes.iloc[i]

print(f"Maximum latitude change below 30° in 6 hours: {max_lat_change_below_30} degrees")
print(f"Maximum longitude change below 30° in 6 hours: {max_lon_change_below_30} degrees")
print(f"Maximum latitude change above 30° in 6 hours: {max_lat_change_above_30} degrees")
print(f"Maximum longitude change above 30° in 6 hours: {max_lon_change_above_30} degrees")
