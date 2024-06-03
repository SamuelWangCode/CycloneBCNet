import torch
from typhoon_intensity_bc.model.model import BCModel
from typhoon_intensity_bc.model.base_method import Base_method


class BCNet(Base_method):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_model(self, **args):
        return BCModel(**self.hparams.model_config)

    def forward(self, x1, x2, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(x1, x2)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(x1, x2)
            pred_y = pred_y[:, -aft_seq_length:]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length

            cur_seq_x1, cur_seq_x2 = x1.clone(), x2.clone()
            for _ in range(d):
                cur_pred = self.model(cur_seq_x1, cur_seq_x2)
                pred_y.append(cur_pred)
                cur_seq_x1, cur_seq_x2 = cur_pred.clone(), cur_seq_x2  # 如果需要，更新 cur_seq_x2
            if m != 0:
                cur_pred = self.model(cur_seq_x1, cur_seq_x2)
                pred_y.append(cur_pred[:, :m])
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def training_step(self, batch, batch_idx):
        batch_x1, batch_x2, batch_y = batch
        pred_y = self(batch_x1, batch_x2)
        batch_y = batch_y[:, -pred_y.shape[1]:, :]
        loss = torch.sqrt(self.criterion(pred_y, batch_y))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
