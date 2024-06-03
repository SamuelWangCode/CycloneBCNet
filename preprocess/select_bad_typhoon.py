import pandas as pd

# 加载数据
data = pd.read_csv('./data_file/error_analysis.csv')

# 设置误差阈值
thresholds = {
    24: 600,
    48: 600,
    72: 1000,
    96: 1800
}

# 筛选不同预报时长超过阈值的个例
filtered_data = data[data.apply(lambda x: x['Max Distance Error (km)'] > thresholds[x['Forecast Hour']], axis=1)]

# 计算有问题的个例的总数和占比
total_issues = len(filtered_data)
total_cases = len(data)
total_id = len(data['Typhoon ID'].unique())
issue_ratio = total_issues / total_cases * 100  # 计算占比百分比

# 获取有问题的台风ID列表
problematic_typhoon_ids = filtered_data['Typhoon ID'].unique()
issue_ratio_typhoon = len(problematic_typhoon_ids) / total_id * 100  # 计算占比百分比
print(f"总问题个例数: {total_issues}")
print(f"问题个例占比: {issue_ratio:.2f}%")
print(f"问题台风ID: {problematic_typhoon_ids}")
print(f"问题台风占比: {issue_ratio_typhoon:.2f}%")