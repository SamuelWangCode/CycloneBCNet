import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# 定义保存分割结果的函数
def save_datasets(train, valid, test, prefix):
    train.to_csv(f'./data_file/{prefix}_train_set.csv', index=False)
    valid.to_csv(f'./data_file/{prefix}_valid_set.csv', index=False)
    test.to_csv(f'./data_file/{prefix}_test_set.csv', index=False)
    print(f"Saved: {prefix} - Training set: {train.shape}, Validation set: {valid.shape}, Test set: {test.shape}")


# 预报时长列表
forecast_hours = [24, 48, 72, 96]

# 加载数据
data = pd.read_csv('./data_file/forecast_instances.csv')
class_counts = data['Class'].value_counts()
print(class_counts)
# 对每个预报时长分别划分数据集
for hours in forecast_hours:
    forecast_data = data[data['Forecast Hour'] == hours]

    # 初始化StratifiedShuffleSplit对象
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    # 分层抽样划分训练集和临时测试集
    for train_index, test_index in split.split(forecast_data, forecast_data['Class']):
        strat_train_set = forecast_data.iloc[train_index]
        strat_test_temp_set = forecast_data.iloc[test_index]

    # 从测试集中进一步划分验证集
    split_val = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=34)
    for train_index, valid_index in split_val.split(strat_test_temp_set, strat_test_temp_set['Class']):
        strat_test_set = strat_test_temp_set.iloc[train_index]
        strat_valid_set = strat_test_temp_set.iloc[valid_index]

    # 保存数据集到CSV文件
    save_datasets(strat_train_set, strat_valid_set, strat_test_set, f'forecast_{hours}h')
