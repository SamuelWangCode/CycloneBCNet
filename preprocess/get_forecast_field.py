import gc
import os
import numpy as np
import time
import datetime

import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
import xarray as xr

from model import build_fields, weaformerV2, ordering


def load_input_from_nc(input_path):
    # pl has dimensions: (longitude: 1440, latitude: 721, level: 13, time: 1)
    # sfc has dimensions: (longitude: 1440, latitude: 721, time: 1)
    pl_path = os.path.join(input_path, "pl.nc")
    sfc_path = os.path.join(input_path, "sfc.nc")
    pl = xr.open_dataset(pl_path).expand_dims('batch', axis=0)
    sfc = xr.open_dataset(sfc_path).expand_dims('batch', axis=0)
    if 'time' in pl.dims:
        pl = pl.isel(time=0)
    if 'time' in sfc.dims:
        sfc = sfc.isel(time=0)
    return build_fields(pl, sfc)  # (1, 73, 721, 1440)


def load_model_and_initial_field(model_wrapper, input_path):
    forecast_model = model_wrapper.load_model()
    all_fields = load_input_from_nc(input_path)
    all_fields = model_wrapper.normalise(all_fields)
    forecast_model = forecast_model.to(model_wrapper.device)
    input_field = all_fields.to(model_wrapper.device, dtype=torch.float32)
    return forecast_model, input_field


def predict(model_wrapper, input_path):
    print('Start predict')
    start_time = time.time()
    forecast_model, initial_field = load_model_and_initial_field(
        model_wrapper,
        input_path=input_path)
    outputs = [initial_field]
    for i in range(int(lead_time / hour_step)):
        output = checkpoint(forecast_model, outputs[i])
        outputs.append(output)
        del output  # 显式删除不再需要的 Tensor
        gc.collect()
        torch.cuda.empty_cache()  # 清空未使用的缓存
    for i in range(len(outputs)):
        outputs[i] = model_wrapper.normalise(outputs[i], reverse=True)
    end_time = time.time()
    print("Predict done in {:.1f} seconds.".format(end_time - start_time))
    return outputs


if __name__ == '__main__':
    root_path = "/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/"
    df = pd.read_csv('./data_file/forecast_instances.csv')
    forecast_tasks = df[['Start Date']].drop_duplicates()  # Remove duplicates to avoid repeated forecasting
    hour_step = 6
    lead_time = 96
    model_wrapper = weaformerV2(root_path=root_path, lead_time=lead_time, hour_step=hour_step)
    print('all sum: ' + str(len(forecast_tasks)) + ' forecast cases.')
    count = 1
    for _, row in forecast_tasks.iterrows():
        print(count)
        start_date = datetime.datetime.strptime((str(row['Start Date'])), "%Y%m%d%H")
        input_path = f"../inputs/{start_date.strftime('%Y%m%dT%H')}/"
        if not os.path.exists(f'./outputs/prediction_{start_date.strftime("%Y%m%dT%H")}.nc'):
            outputs = predict(model_wrapper, input_path)
            output_np = torch.stack(outputs).cpu().numpy().squeeze()

            # 时间步定义
            time_steps = pd.date_range(start=start_date, periods=output_np.shape[0], freq='6H')

            # 创建 Dataset 而不是 DataArray
            coords = {
                "time": time_steps,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 359.75, 1440)
            }

            data_vars = {}
            for i, var in enumerate(ordering):
                data_vars[var] = (("time", "lat", "lon"), output_np[:, i, :, :])

            dataset = xr.Dataset(data_vars, coords=coords)

            # 文件保存路径
            output_filename = f'./outputs/prediction_{start_date.strftime("%Y%m%dT%H")}.nc'
            dataset.to_netcdf(output_filename)
            print(f"Saved prediction to {output_filename}")
            count = count + 1
        else:
            count = count + 1
            print(f'./outputs/prediction_{start_date.strftime("%Y%m%dT%H")}.nc exists.')
