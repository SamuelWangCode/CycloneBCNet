import os

from captum.attr import LayerGradCam, FeatureAblation, visualization as viz
import matplotlib.pyplot as plt
import pandas as pd
import torch
import xarray as xr
import numpy as np

from typhoon_intensity_bc.project.construct_dataset import load_model, MultiChannelNormalizer, FieldNormalizer, \
    model_params_track_96, model_params_96
from haversine import haversine, Unit


def model_forward_wrapper(field_input, position):
    outputs = track_model(field_input, position)
    result = 0
    for i in range(outputs.shape[1]):
        result += outputs[:, i, :]
    return result  # 选择最后一个时间步的输出


def intensity_model_forward_wrapper(field_input, value):
    outputs = intensity_model(field_input, value)
    result = 0
    for i in range(outputs.shape[1]):
        result += outputs[:, i, :]
    return result  # 选择最后一个时间步的输出


def plot_combined(cam, intensity_cam, forecast_var, u_wind, v_wind, t500, t850, t1000, lat_grid, lon_grid, time_step):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 台风路径订正的CAM图像
    track_cam_img = cam[time_step, :, :].cpu().detach().numpy()
    im1 = axes[0].imshow(track_cam_img, cmap='jet', aspect='auto', interpolation='bilinear', origin='lower',
                            extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
    cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.1)
    cbar1.set_label('Grad-CAM Intensity', labelpad=20)
    axes[0].set_title(f'Track Correction Grad-CAM at Timestep {time_step}', fontsize=24)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    # # 提取真实台风路径中的经纬度并转换为实际值
    # real_lats = [item[1] / 10 for item in real_path]
    # real_lons = [item[2] / 10 for item in real_path]
    #
    # # 提取预报台风路径中的经纬度
    # forecast_lons, forecast_lats = forecast_path[:, 0], forecast_path[:, 1]
    #
    # # 在图上叠加真实台风路径
    # axes[0, 0].plot(real_lons, real_lats, 'k-', marker='o', markersize=4, markerfacecolor='black',
    #                 markeredgecolor='black', label='Real Path')
    #
    # # 在图上叠加预报台风路径
    # axes[0, 0].plot(forecast_lons, forecast_lats, 'b--', marker='s', markersize=4, markerfacecolor='blue',
    #                 markeredgecolor='blue', label='Forecast Path')
    #
    # # 当前时间步的台风中心位置
    # current_lat = real_lats[time_step]
    # current_lon = real_lons[time_step]
    #
    # # 标记当前时间步的台风中心位置
    # axes[0, 0].plot(current_lon, current_lat, 'ro', markersize=8, markerfacecolor='red', markeredgecolor='red')

    # 台风强度订正的CAM图像
    intensity_cam_img = intensity_cam[time_step, :, :].cpu().detach().numpy()
    im2 = axes[1].imshow(intensity_cam_img, cmap='jet', aspect='auto', interpolation='bilinear', origin='lower',
                            extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
    cbar2 = plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.1)
    cbar2.set_label('Grad-CAM Intensity', labelpad=20)
    axes[1].set_title(f'Intensity Correction Grad-CAM at Timestep {time_step}', fontsize=24)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    # # 提取真实台风路径中的经纬度并转换为实际值
    # real_lats = [item[1] / 10 for item in real_path]
    # real_lons = [item[2] / 10 for item in real_path]
    #
    # # 提取预报台风路径中的经纬度
    # forecast_lons, forecast_lats = forecast_path[:, 0], forecast_path[:, 1]
    #
    # # 在图上叠加真实台风路径
    # axes[0, 1].plot(real_lons, real_lats, 'k-', marker='o', markersize=4, markerfacecolor='black',
    #                 markeredgecolor='black', label='Real Path')
    #
    # # 在图上叠加预报台风路径
    # axes[0, 1].plot(forecast_lons, forecast_lats, 'b--', marker='s', markersize=4, markerfacecolor='blue',
    #                 markeredgecolor='blue', label='Forecast Path')
    #
    # # 当前时间步的台风中心位置
    # current_lat = real_lats[time_step]
    # current_lon = real_lons[time_step]
    #
    # # 标记当前时间步的台风中心位置
    # axes[0, 1].plot(current_lon, current_lat, 'ro', markersize=8, markerfacecolor='red', markeredgecolor='red')

    # # 预报场图像
    # forecast_img = forecast_var.squeeze()
    # wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)
    # # 绘制风速填色
    # wind_speed_plot = axes[0, 2].contourf(wind_speed, cmap='Blues', alpha=1,
    #                                       extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
    #
    # # 添加风速的colorbar
    # cbar3 = plt.colorbar(wind_speed_plot, ax=axes[0, 2], orientation='horizontal', pad=0.08)
    # cbar3.set_label(f'Wind Speed (m s$^-1$)', labelpad=20)
    # # 计算并绘制500公里内的平均风速方向
    # center_lat, center_lon = lat_grid[forecast_img.shape[0] // 2, forecast_img.shape[1] // 2], lon_grid[
    #     forecast_img.shape[0] // 2, forecast_img.shape[1] // 2]
    # u_wind_flat, v_wind_flat, lat_flat, lon_flat = u_wind.flatten(), v_wind.flatten(), lat_grid.flatten(), lon_grid.flatten()
    # distances = np.array(
    #     [haversine((center_lat, center_lon), (lat, lon), unit=Unit.KILOMETERS) for lat, lon in zip(lat_flat, lon_flat)])
    # mask = distances <= 500
    # mean_u = np.mean(u_wind_flat[mask])
    # mean_v = np.mean(v_wind_flat[mask])
    # axes[0, 2].quiver(center_lon, center_lat, [mean_u], [mean_v], scale=20, color='red')
    # circle = plt.Circle((center_lon, center_lat), 500 / 111, color='red', linestyle='--', fill=False, linewidth=1.5)
    # axes[0, 2].add_patch(circle)
    # axes[0, 2].set_title(f'TianXing Forecast for Wind mean', fontsize=24)
    # axes[0, 2].set_xlabel('Longitude')
    # axes[0, 2].set_ylabel('Latitude')
    # # 提取真实台风路径中的经纬度并转换为实际值
    # real_lats = [item[1] / 10 for item in real_path]
    # real_lons = [item[2] / 10 for item in real_path]
    #
    # # # 提取预报台风路径中的经纬度
    # # forecast_lons, forecast_lats = forecast_path[:, 0], forecast_path[:, 1]
    # #
    # # # 在图上叠加真实台风路径
    # # axes[0, 2].plot(real_lons, real_lats, 'k-', marker='o', markersize=4, markerfacecolor='black',
    # #                 markeredgecolor='black', label='Real Path')
    # #
    # # # 在图上叠加预报台风路径
    # # axes[0, 2].plot(forecast_lons, forecast_lats, 'b--', marker='s', markersize=4, markerfacecolor='blue',
    # #                 markeredgecolor='blue', label='Forecast Path')
    # #
    # # # 当前时间步的台风中心位置
    # # current_lat = real_lats[time_step]
    # # current_lon = real_lons[time_step]
    # #
    # # # 标记当前时间步的台风中心位置
    # # axes[0, 2].plot(current_lon, current_lat, 'ro', markersize=8, markerfacecolor='red', markeredgecolor='red')
    #
    # # 绘制温度场
    # im_temp = axes[1, 0].contourf(t500 - t500.mean(), cmap='coolwarm',
    #                               extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
    # cbar_temp = plt.colorbar(im_temp, ax=axes[1, 0], orientation='horizontal', pad=0.08)
    # cbar_temp.set_label(f'Temperature at 500hPa (K)', labelpad=20)
    # axes[1, 0].set_title(f'Temperature at 500hPa', fontsize=24)
    # axes[1, 0].set_xlabel('Longitude')
    # axes[1, 0].set_ylabel('Latitude')
    # # # 提取真实台风路径中的经纬度并转换为实际值
    # # real_lats = [item[1] / 10 for item in real_path]
    # # real_lons = [item[2] / 10 for item in real_path]
    # #
    # # # 提取预报台风路径中的经纬度
    # # forecast_lons, forecast_lats = forecast_path[:, 0], forecast_path[:, 1]
    # #
    # # # 在图上叠加真实台风路径
    # # axes[1, 0].plot(real_lons, real_lats, 'k-', marker='o', markersize=4, markerfacecolor='black',
    # #                 markeredgecolor='black', label='Real Path')
    # #
    # # # 在图上叠加预报台风路径
    # # axes[1, 0].plot(forecast_lons, forecast_lats, 'b--', marker='s', markersize=4, markerfacecolor='blue',
    # #                 markeredgecolor='blue', label='Forecast Path')
    # #
    # # # 当前时间步的台风中心位置
    # # current_lat = real_lats[time_step]
    # # current_lon = real_lons[time_step]
    # #
    # # # 标记当前时间步的台风中心位置
    # # axes[1, 0].plot(current_lon, current_lat, 'ro', markersize=8, markerfacecolor='red', markeredgecolor='red')
    #
    # # 绘制温度场
    # im_temp = axes[1, 1].contourf(t850 - t850.mean(), cmap='coolwarm',
    #                               extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
    # cbar_temp = plt.colorbar(im_temp, ax=axes[1, 1], orientation='horizontal', pad=0.08)
    # cbar_temp.set_label(f'Temperature at 850hPa (K)', labelpad=20)
    # axes[1, 1].set_title(f'Temperature at 850hPa', fontsize=24)
    # axes[1, 1].set_xlabel('Longitude')
    # axes[1, 1].set_ylabel('Latitude')
    # # # 提取真实台风路径中的经纬度并转换为实际值
    # # real_lats = [item[1] / 10 for item in real_path]
    # # real_lons = [item[2] / 10 for item in real_path]
    # #
    # # # 提取预报台风路径中的经纬度
    # # forecast_lons, forecast_lats = forecast_path[:, 0], forecast_path[:, 1]
    # #
    # # # 在图上叠加真实台风路径
    # # axes[1, 1].plot(real_lons, real_lats, 'k-', marker='o', markersize=4, markerfacecolor='black',
    # #                 markeredgecolor='black', label='Real Path')
    # #
    # # # 在图上叠加预报台风路径
    # # axes[1, 1].plot(forecast_lons, forecast_lats, 'b--', marker='s', markersize=4, markerfacecolor='blue',
    # #                 markeredgecolor='blue', label='Forecast Path')
    # #
    # # # 当前时间步的台风中心位置
    # # current_lat = real_lats[time_step]
    # # current_lon = real_lons[time_step]
    # #
    # # # 标记当前时间步的台风中心位置
    # # axes[1, 1].plot(current_lon, current_lat, 'ro', markersize=8, markerfacecolor='red', markeredgecolor='red')
    #
    # # 绘制温度场
    # im_temp = axes[1, 2].contourf(t1000 - t1000.mean(), cmap='coolwarm',
    #                               extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
    # cbar_temp = plt.colorbar(im_temp, ax=axes[1, 2], orientation='horizontal', pad=0.08)
    # cbar_temp.set_label(f'Temperature at 1000hPa (K)', labelpad=20)
    # axes[1, 2].set_title(f'Temperature at 1000hPa', fontsize=24)
    # axes[1, 2].set_xlabel('Longitude')
    # axes[1, 2].set_ylabel('Latitude')
    # # # 提取真实台风路径中的经纬度并转换为实际值
    # # real_lats = [item[1] / 10 for item in real_path]
    # # real_lons = [item[2] / 10 for item in real_path]
    # #
    # # # 提取预报台风路径中的经纬度
    # # forecast_lons, forecast_lats = forecast_path[:, 0], forecast_path[:, 1]
    # #
    # # # 在图上叠加真实台风路径
    # # axes[1, 2].plot(real_lons, real_lats, 'k-', marker='o', markersize=4, markerfacecolor='black',
    # #                 markeredgecolor='black', label='Real Path')
    # #
    # # # 在图上叠加预报台风路径
    # # axes[1, 2].plot(forecast_lons, forecast_lats, 'b--', marker='s', markersize=4, markerfacecolor='blue',
    # #                 markeredgecolor='blue', label='Forecast Path')
    # #
    # # # 当前时间步的台风中心位置
    # # current_lat = real_lats[time_step]
    # # current_lon = real_lons[time_step]
    # #
    # # # 标记当前时间步的台风中心位置
    # # axes[1, 2].plot(current_lon, current_lat, 'ro', markersize=8, markerfacecolor='red', markeredgecolor='red')

    plt.tight_layout()
    plt.savefig(f'./evaluate/case_study/combined_pic_{time_step}.svg', dpi=600, bbox_inches='tight')
    plt.close()


def trim_and_interpolate_data(data_array, mask=None):
    if mask is not None:
        data_array = np.where(mask, data_array, np.nan)
    # 获取有效数据区域的索引
    valid_data = ~np.isnan(data_array)
    coords = np.argwhere(valid_data)
    if coords.size == 0:
        raise ValueError("No valid data found")

    # 获取边界
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # +1 因为索引从0开始

    # 裁剪数据
    trimmed_data = data_array[y_min:y_max, x_min:x_max]

    return trimmed_data, y_min, y_max, x_min, x_max


def convert_geopotential_to_height(geopotential):
    g = 9.80665  # 地球重力加速度
    geopotential_height = geopotential / g
    return geopotential_height


# Example usage
# 实例化训练模块
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
track_model = load_model('./data_file/BCNet/typhoon_track_96h/epoch=3991-step=3992.ckpt', device,
                         model_params_track_96)
intensity_model = load_model('./data_file/BCNet/typhoon_intensity_96h/epoch=2034-step=2035.ckpt', device,
                             model_params_96)
position_normalizer = MultiChannelNormalizer(num_channels=2)
position_normalizer.load('./data_file/stats', 'position')
intensity_normalizer = MultiChannelNormalizer(num_channels=2)
intensity_normalizer.load('./data_file/stats', 'intensity')
field_normalizer_24 = FieldNormalizer(num_channels=73)
field_normalizer_24.load('./data_file/stats', 'field_24')
field_normalizer_48 = FieldNormalizer(num_channels=73)
field_normalizer_48.load('./data_file/stats', 'field_48')
field_normalizer_72 = FieldNormalizer(num_channels=73)
field_normalizer_72.load('./data_file/stats', 'field_72')
field_normalizer_96 = FieldNormalizer(num_channels=73)
field_normalizer_96.load('./data_file/stats', 'field_96')
field_normalizer = field_normalizer_96
root_dir_field = './evaluate/case_study/'
root_dir_value = './evaluate/case_study/'
root_dir_target = './evaluate/case_study/'
root_dir_track = './evaluate/case_study/'
test_data_set = './evaluate/case_study/forecast_instances.csv'
data_info = pd.read_csv(test_data_set)
for index, row in data_info.iterrows():
    field_path = os.path.join(root_dir_field, f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}.pt")
    value_input_path = os.path.join(root_dir_value,
                                    f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_input.pt")
    target_path = os.path.join(root_dir_target,
                               f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}_target.pt")
    field_data = torch.load(field_path)
    value_input = torch.load(value_input_path)
    target = torch.load(target_path)
    field_input = torch.tensor(field_normalizer.transform(field_data)).unsqueeze(0)  # (1,17,73,73,73)
    # track
    track_input = value_input[:, :2]
    target = target[:, :2]
    position = torch.tensor(position_normalizer.transform(track_input))
    position = position.clone().detach().unsqueeze(0)  # (1,17,2)
    corrected_position = track_model(field_input, position)
    corrected_position[0] = position[0]
    intensity_input = torch.tensor(intensity_normalizer.transform(value_input[:, 2:])).clone().detach().unsqueeze(0)
    value = torch.cat((corrected_position, intensity_input), dim=2)
    layer_gc = LayerGradCam(model_forward_wrapper, track_model.encoder1)
    intensity_layer_gc = LayerGradCam(intensity_model_forward_wrapper, intensity_model.encoder1)
    # 计算归因
    channel_attributions = []
    cam = layer_gc.attribute((field_input, position), target=0, attribute_to_layer_input=True)  # (17,1,73,73)
    cam += layer_gc.attribute((field_input, position), target=1, attribute_to_layer_input=True)
    intensity_cam = intensity_layer_gc.attribute((field_input, value), target=0, attribute_to_layer_input=True)
    intensity_cam += intensity_layer_gc.attribute((field_input, value), target=1, attribute_to_layer_input=True)
    attributions = torch.reshape(cam, (17, 73, 73))
    intensity_attributions = torch.reshape(intensity_cam, (17, 73, 73))
    # 读取预报场数据
    forecast_data = xr.open_dataset('/data4/wxz_data/typhoon_intensity_bc/field_data_extraction/2306_2023073000_96.nc')
    # 获取经纬度网格
    var = ['z500']
    real_path = [
        [4, 187, 1328], [4, 195, 1327], [5, 203, 1324], [5, 211, 1322], [5, 221, 1320],
        [6, 228, 1315], [6, 234, 1311], [6, 240, 1303], [6, 246, 1294], [6, 250, 1287],
        [6, 253, 1280], [6, 255, 1274], [6, 257, 1268], [5, 260, 1261], [5, 262, 1256],
        [5, 265, 1250], [5, 266, 1247]
    ]
    forecast_path = np.array([
        [132.8000, 18.7000],
        [132.7500, 19.7500],
        [132.5000, 20.7500],
        [132.0000, 21.7500],
        [131.7500, 22.5000],
        [131.0000, 23.5000],
        [130.2500, 24.2500],
        [129.2500, 25.0000],
        [128.2500, 25.5000],
        [127.2500, 26.0000],
        [126.5000, 26.0000],
        [125.7500, 26.2500],
        [125.2500, 26.5000],
        [124.7500, 26.7500],
        [124.0000, 27.0000],
        [123.5000, 27.2500],
        [123.0000, 27.5000]
    ])
    for t in range(attributions.shape[0]):
        # if t == 9:
            for v in var:
                if v == 'z500':
                    forecast_var = convert_geopotential_to_height(forecast_data[v].isel(time=t).values)
                else:
                    forecast_var = forecast_data[v].isel(time=t).values
                # 创建掩码，只有有数据的格点为1，其他为0
                trimmed, y_min, y_max, x_min, x_max = trim_and_interpolate_data(forecast_var)
                # 获取原始的经纬度数据
                lat_full = forecast_data['lat'].values
                lon_full = forecast_data['lon'].values

                # 使用meshgrid创建全网格的经纬度
                lat_grid_full, lon_grid_full = np.meshgrid(lat_full, lon_full, indexing='ij')

                # 使用trim的索引来裁剪经纬度网格
                lat_grid_trimmed = lat_grid_full[y_min:y_max, x_min:x_max]
                lon_grid_trimmed = lon_grid_full[y_min:y_max, x_min:x_max]
                u_wind = (75 * forecast_data['u300'] + 100 * forecast_data['u400'] + 150 * forecast_data['u500'] + 175 *
                          forecast_data['u600'] + 175 * forecast_data['u700'] + 150 * forecast_data['u850']) / 825
                v_wind = (75 * forecast_data['v300'] + 100 * forecast_data['v400'] + 150 * forecast_data['v500'] + 175 *
                          forecast_data['v600'] + 175 * forecast_data['v700'] + 150 * forecast_data['v850']) / 825
                u_wind_trimmed = u_wind.isel(time=t).values[y_min:y_max, x_min:x_max]
                v_wind_trimmed = v_wind.isel(time=t).values[y_min:y_max, x_min:x_max]
                t500_trimmed = forecast_data['t500'].isel(time=t).values[y_min:y_max, x_min:x_max]
                t850_trimmed = forecast_data['t850'].isel(time=t).values[y_min:y_max, x_min:x_max]
                t1000_trimmed = forecast_data['t1000'].isel(time=t).values[y_min:y_max, x_min:x_max]
                plot_combined(attributions, intensity_attributions, trimmed, u_wind_trimmed, v_wind_trimmed,
                              t500_trimmed,
                              t850_trimmed, t1000_trimmed, lat_grid_trimmed, lon_grid_trimmed, t)

    # selected_channels = [41]
    # feature_ablation = FeatureAblation(model_forward_wrapper)
    #
    # # 计算特征消融
    # ablated_attributions = feature_ablation.attribute(
    #     (field_input, position),
    #     target=0,  # 或者设置为你想要评估的目标
    #     perturbations_per_eval=1,  # 确保这是一个有效的整数
    #     n_steps=50  # 适当减少步骤以提高速度
    # )
    #
    # # 将特征消融结果转化为 numpy 数组
    # ablated_attributions_np = ablated_attributions.cpu().detach().numpy()
    #
    # # 可视化特征消融结果
    # print("Feature Ablation attributions:", ablated_attributions_np)
    #
    # # 特征名称列表
    # features = [f'Channel {i}' for i in selected_channels]
    #
    # # 计算每个特征的重要性（取绝对值的均值）
    # feature_importance = np.mean(np.abs(ablated_attributions_np), axis=(0, 1))
    #
    # # 可视化特征重要性
    # plt.figure(figsize=(15, 8))
    # plt.bar(features, feature_importance)
    # plt.xlabel('Features')
    # plt.ylabel('Importance')
    # plt.title('Feature Importance from Feature Ablation')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig('./evaluate/case_study/feature_ablation_importance.png', dpi=600)
    # plt.show()
