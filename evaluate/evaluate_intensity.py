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
        results = []
        t_test_results = []
        # 绘制误差分布图
        data = pd.read_csv(f'./data_file/forecast_diff/typhoon_intensity_{forecast_time}h/intensity_data.csv')
        # 计算统计指标
        for dimension in ['Vmax', 'Pmin']:
            if dimension == 'Vmax':
                units = 'm s$^{-1}$'
                limit = 40
            else:
                units = 'hPa'
                limit = 70
            true_values = data[f'{dimension} True']
            origin_values = data[f'{dimension} Origin']
            corrected_values = data[f'{dimension} Corrected']
            # 绘制误差分布图
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.hist(abs(true_values - origin_values), bins=30, alpha=0.7, label='TianXing')
            plt.hist(abs(true_values - corrected_values), bins=30, alpha=0.7, label='CycloneBCNet')
            plt.xlim([0, limit])  # 设置x轴的范围
            plt.title(f'{dimension} Error Distribution ({forecast_time}h)')
            plt.xlabel(f'Error ({units})')
            plt.ylabel('Frequency')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(abs(true_values - origin_values), abs(true_values - corrected_values), alpha=0.6)
            plt.title(f'{dimension} Error Comparison ({forecast_time}h)')
            plt.xlabel(f'TianXing Error ({units})')
            plt.ylabel(f'CycloneBCNet Error ({units})')
            plt.xlim([0, limit])  # 设置x轴的范围
            plt.ylim([0, limit])  # 设置x轴的范围
            plt.axline((0, 0), slope=1, color="red", linestyle="--")  # 添加y=x参考线

            plt.tight_layout()
            plt.savefig(
                f'./data_file/forecast_diff/typhoon_intensity_{forecast_time}h/error_comparison_plots_{dimension}.svg',
                dpi=600,
                bbox_inches='tight')
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1行2列的子图
        for i, dimension in enumerate(['Vmax', 'Pmin']):
            if dimension == 'Vmax':
                units = 'm s$^{-1}$'
            else:
                units = 'hPa'
            true_values = data[f'{dimension} True']
            origin_values = data[f'{dimension} Origin']
            corrected_values = data[f'{dimension} Corrected']
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
            ax.set_xlabel(f'CMA Best-track {dimension} ({units})')
            ax.set_ylabel(f'Forecasted {dimension} ({units})')
            ax.legend()
            ax.grid(True)
        # 调整布局
        plt.tight_layout()
        plt.savefig(f'./data_file/forecast_diff/typhoon_intensity_{forecast_time}h/scatter_intensity.svg',
                    dpi=600,
                    bbox_inches='tight')

        df_results = pd.DataFrame(results)
        df_t_test = pd.DataFrame(t_test_results)
        # 将所有结果保存到CSV文件
        df_results.to_csv(f'./data_file/forecast_diff/typhoon_intensity_{forecast_time}h/statistical_results.csv',
                          index=False)
        df_t_test.to_csv(f'./data_file/forecast_diff/typhoon_intensity_{forecast_time}h/t_test_results.csv', index=False)

        # 打印结果以便查看
        print(df_results)
        print(df_t_test)
