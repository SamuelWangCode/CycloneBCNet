import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_statistics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mbe = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    max_error = np.max(np.abs(y_pred - y_true))
    min_error = np.min(np.abs(y_pred - y_true))
    return mae, rmse, mbe, r2, max_error, min_error


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    for forecast_time in [24, 48, 72, 96]:
        origin_distance_arr = np.load(
            f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/origin_distance_arr.npy')
        forecast_distance_arr = np.load(
            f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/forecast_distance_arr.npy')
        # 计算平均误差
        mean_origin_error = np.mean(origin_distance_arr, axis=1)
        mean_forecast_error = np.mean(forecast_distance_arr, axis=1)
        # 进行配对的t检验
        t_stat, p_value = ttest_rel(mean_origin_error, mean_forecast_error)
        t_test_results = []
        t_test_results.append({
            'Metric': 'Distance - Original',
            'T-statistic': t_stat,
            'P-value': p_value
        })
        results = []
        # 绘制误差分布图
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(mean_origin_error, bins=30, alpha=0.7, label='TianXing')
        plt.hist(mean_forecast_error, bins=30, alpha=0.7, label='CycloneBCNet')
        plt.xlim([0, 400])  # 设置x轴的范围
        plt.title(f'Track Position Error Distribution ({forecast_time}h)')
        plt.xlabel('Distance Error (km)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(mean_origin_error, mean_forecast_error, alpha=0.6)
        plt.title(f'Track Position Error Comparison ({forecast_time}h)')
        plt.xlabel('TianXing Error (km)')
        plt.ylabel('CycloneBCNet Error (km)')
        plt.xlim([0, 400])  # 设置x轴的范围
        plt.ylim([0, 400])  # 设置y轴的范围
        plt.axline((0, 0), slope=1, color="red", linestyle="--")  # 添加y=x参考线

        plt.tight_layout()
        plt.savefig(f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/error_comparison_plots_distance.svg',
                    dpi=600, bbox_inches='tight')

        data = pd.read_csv(f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/position_data.csv')
        # 计算统计指标
        # 设置图形大小和子图
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1行2列的子图
        for i, dimension in enumerate(['Latitude', 'Longitude']):
            true_values = data[f'{dimension} True']
            origin_values = data[f'{dimension} Origin']
            corrected_values = data[f'{dimension} Corrected']
            # 绘制误差分布图
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # plt.hist(abs(true_values - origin_values), bins=30, alpha=0.7, label='Original Error')
            # plt.hist(abs(true_values - corrected_values), bins=30, alpha=0.7, label='Correct Error')
            # plt.title('Error Distribution')
            # plt.xlabel('Error')
            # plt.ylabel('Frequency')
            # plt.legend()
            #
            # plt.subplot(1, 2, 2)
            # plt.scatter(abs(true_values - origin_values), abs(true_values - corrected_values), alpha=0.6)
            # plt.title('Error Comparison')
            # plt.xlabel('Original Error')
            # plt.ylabel('Correct Error')
            # plt.axline((0, 0), slope=1, color="red", linestyle="--")  # 添加y=x参考线
            #
            # plt.tight_layout()
            # plt.savefig(
            #     f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/error_comparison_plots_{dimension}.svg',
            #     dpi=600, bbox_inches='tight')
            # 对于原始预报和真值的 t 统计和 p 值
            t_stat_origin, p_value_origin = ttest_rel(true_values, origin_values)
            # 对于订正后预报和真值的 t 统计和 p 值
            t_stat_corrected, p_value_corrected = ttest_rel(true_values, corrected_values)

            t_test_results.append({
                'Metric': f'{dimension} - Original',
                'T-statistic': t_stat_origin,
                'P-value': p_value_origin
            })
            t_test_results.append({
                'Metric': f'{dimension} - Corrected',
                'T-statistic': t_stat_corrected,
                'P-value': p_value_corrected
            })
            # 对于原始预报
            stats_orig = calculate_statistics(true_values, origin_values)
            # 对于订正后预报
            stats_corr = calculate_statistics(true_values, corrected_values)
            # 保存结果
            results.append({
                'Metric': f'{dimension} - Original',
                'MAE': stats_orig[0],
                'RMSE': stats_orig[1],
                'MBE': stats_orig[2],
                'R2': stats_orig[3],
                'Max Error': stats_orig[4],
                'Min Error': stats_orig[5]
            })

            results.append({
                'Metric': f'{dimension} - Corrected',
                'MAE': stats_corr[0],
                'RMSE': stats_corr[1],
                'MBE': stats_corr[2],
                'R2': stats_corr[3],
                'Max Error': stats_corr[4],
                'Min Error': stats_corr[5]
            })

            # 在子图上绘制散点图
            ax = axs[i]
            ax.scatter(true_values, origin_values, alpha=0.5, marker='o', label='TianXing')
            ax.scatter(true_values, corrected_values, alpha=0.5, marker='^', label='CycloneBCNet')
            ax.plot(true_values, true_values, 'r--')  # 添加 y=x 参考线
            ax.set_title(f'{dimension} ({forecast_time}h)')
            ax.set_xlabel(f'CMA Best-track {dimension} (°)')
            ax.set_ylabel(f'Forecasted {dimension} (°)')
            ax.legend()
            ax.grid(True)
        # 调整布局
        plt.tight_layout()

        # 保存图形
        plt.savefig(f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/scatter_lon_lat.svg',
                    dpi=600,
                    bbox_inches='tight')

        # 计算距离的统计指标
        origin_distances = np.sqrt(
            (data['Latitude Origin'] - data['Latitude True']) ** 2 + (
                    data['Longitude Origin'] - data['Longitude True']) ** 2)
        corrected_distances = np.sqrt((data['Latitude Corrected'] - data['Latitude True']) ** 2 + (
                data['Longitude Corrected'] - data['Longitude True']) ** 2)
        true_distances = np.zeros_like(origin_distance_arr)

        # 统计原始数据误差
        stats_orig = calculate_statistics(true_distances, origin_distance_arr)
        # 统计订正后数据误差
        stats_corr = calculate_statistics(true_distances, forecast_distance_arr)
        results.append({
            'Metric': f'distance - Original',
            'MAE': stats_orig[0],
            'RMSE': stats_orig[1],
            'MBE': stats_orig[2],
            'R2': stats_orig[3],
            'Max Error': stats_orig[4],
            'Min Error': stats_orig[5]
        })
        results.append({
            'Metric': f'distance - Corrected',
            'MAE': stats_corr[0],
            'RMSE': stats_corr[1],
            'MBE': stats_corr[2],
            'R2': stats_corr[3],
            'Max Error': stats_corr[4],
            'Min Error': stats_corr[5]
        })
        df_results = pd.DataFrame(results)
        df_t_test = pd.DataFrame(t_test_results)
        # 将所有结果保存到CSV文件
        df_results.to_csv(f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/statistical_results.csv',
                          index=False)
        df_t_test.to_csv(f'./data_file/forecast_diff/typhoon_track_{forecast_time}h/t_test_results.csv', index=False)

        # 打印结果以便查看
        print(df_results)
        print(df_t_test)
