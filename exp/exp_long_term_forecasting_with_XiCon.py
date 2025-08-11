from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, shape_metric
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import sys
import os
import time
import warnings
import numpy as np

from scipy.stats import rankdata, norm
from statsmodels.tsa.api import acf
from layers.losses import AutoCon, XiCon

from multiprocessing import Manager, Process, cpu_count

warnings.filterwarnings('ignore')

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        l1_loss = self.l1_loss(outputs, targets)
        l2_loss = self.l2_loss(outputs, targets)

        loss = self.alpha * l1_loss + self.beta * l2_loss
        return loss

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Exp_Long_Term_Forecast_with_XiCon(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_with_XiCon, self).__init__(args)

        self.XiCon = args.XiCon
        self.AutoCon_lambda = args.AutoCon_lambda
        st = time.time()
        self.XiCon_loss = self.init_XiCon(args)
        ed = time.time()
        print(f'Autocorrelation calculation time: {ed-st:.4f}')

    def init_XiCon(self, args):
        #_xicor =self._xicor
        target_data, _ = self._get_data(flag='train')
        target_data = target_data.data_x.copy()
        smoother = series_decomp(args.seq_len+1)
        x = torch.from_numpy(target_data).unsqueeze(0)
        _, target_data = smoother(x)
        target_data = target_data.squeeze(0).numpy()
        acf_values = []
        for i_ch in range(target_data.shape[-1]):
            acf_values.append(acf(target_data[..., i_ch], nlags=len(target_data)))
        patching = args.patching
        patch_len = args.patch_len
        stride = args.stride
        nlags=len(target_data)
        manager = Manager()
        return_dict = manager.dict()

        xicor_values = []
        def cal_xicor(target_data, pre, return_dict):
            tmp = []
            process_length = int(len(target_data) / self.worker)
            if pre != (self.worker -1):
                range_nlags = range(len(target_data))[(pre * process_length):((pre + 1) * process_length)]
            else:
                range_nlags = range(len(target_data))[((pre) * process_length):]
            for i_ch in range(target_data.shape[-1]):
                channel_data = target_data[:, i_ch]
                channel_xicor = []
                
                for lag in range_nlags:
                    if lag == 0:
                        statistic, _ = self._xicor(channel_data, channel_data)
                    else:
                        statistic, _ = self._xicor(channel_data[:-lag], channel_data[lag:])
                    channel_xicor.append(statistic)
                
                tmp.append(channel_xicor)


            return_dict[pre] = tmp

        if args.cpu_worker == None:
            self.worker = cpu_count()
        else:
            if type(args.cpu_worker) == int:
                self.worker = args.cpu_worker
            else:
                raise Exception("cpu_worker must have integer or None")
        
        print('number of available CPU: ',self.worker)

        for i in range(self.worker):
            globals()[f'xicor_process{i}'] = Process(target=cal_xicor, args=(target_data, i, return_dict))
              
        for i in range(self.worker):
            globals()[f'xicor_process{i}'].start() 
        
        for i in range(self.worker):
            globals()[f'xicor_process{i}'].join()

        for j in range(target_data.shape[-1]):
            tmp = []
            for i in range(self.worker):
                tmp.extend(return_dict[i][j])
            
            xicor_values.append(tmp)


        if self.args.model == 'XiCon_PatchMixer':
            acf_values = np.stack(acf_values, axis=0)
            xicor_values = np.stack(xicor_values, axis=0)
            loss = XiCon(args.batch_size, args.seq_len, np.abs(acf_values), xicor_values, patch_len, stride , temperature=1.0, base_temperature=1.0, omega = args.omega)
            print(f'Auto-correlation values(abs):{acf_values[0, :2]} ~ {acf_values[0, -2:]}')
            print(f'Xi-correlation values:{xicor_values[0, :2]} ~ {xicor_values[0, -2:]}')
        elif self.args.model == 'XiCon':
            acf_values = np.stack(acf_values, axis=0)
            xicor_values = np.stack(xicor_values, axis=0)
            loss = XiCon(args.batch_size, args.seq_len, np.abs(acf_values), xicor_values, patch_len, stride , temperature=1.0, base_temperature=1.0, omega = args.omega)
            print(f'Auto-correlation values(abs):{acf_values[0, :2]} ~ {acf_values[0, -2:]}')
            print(f'Xi-correlation values:{xicor_values[0, :2]} ~ {xicor_values[0, -2:]}')
        elif self.args.model == 'XiCon_PatchTST':
            acf_values = np.stack(acf_values, axis=0)
            xicor_values = np.stack(xicor_values, axis=0)
            loss = XiCon(args.batch_size, args.seq_len, np.abs(acf_values), xicor_values, patch_len, stride , temperature=1.0, base_temperature=1.0, omega = args.omega)
            print(f'Auto-correlation values(abs):{acf_values[0, :2]} ~ {acf_values[0, -2:]}')
            print(f'Xi-correlation values:{xicor_values[0, :2]} ~ {xicor_values[0, -2:]}')       
        else:
            acf_values = np.stack(acf_values, axis=0).mean(axis=0)
            xicor_values = np.stack(xicor_values, axis=0)
            loss = AutoCon(args.batch_size, args.seq_len, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
            print(f'Auto-correlation values(abs):{acf_values[:2]} ~ {acf_values[-2:]}')
        return loss

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(f'model parameters:{self.count_parameters(model)}')
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        if self.args.loss_flag==1: # loss_flag 0 for MSE, 1 for MAE, 2 for both of MSE & MAE, 3 for SmoothL1loss
            criterion = nn.L1Loss()
        elif self.args.loss_flag==2:
            criterion = MultiTaskLoss(alpha=0.5, beta=0.5)
        elif self.args.loss_flag == 3:
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    """
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    """
    
    def _xicor(self, x, y, ties="auto"):
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        n = len(y) # 8640
        
        if len(x) != n:
            raise IndexError(
                f"x, y length mismatch: {len(x)}, {len(y)}"
            )

        if ties == "auto":
            ties = len(np.unique(y)) < n
        elif not isinstance(ties, bool):
            raise ValueError(
                f"expected ties either \"auto\" or boolean, "
                f"got {ties} ({type(ties)}) instead"
            )
        
        y = y[np.argsort(x)]
        r = rankdata(y, method="ordinal")
        nominator = np.sum(np.abs(np.diff(r)))

        if ties:
            l = rankdata(y, method="max")
            denominator = 2 * np.sum(l * (n - l))
            nominator *= n
        else:
            denominator = np.power(n, 2) - 1
            nominator *= 3

        if denominator == 0:
            denominator = 1e-8

        statistic = 1 - nominator / denominator  # upper bound is (n - 2) / (n + 1)
        p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

        return statistic, p_value


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if not self.XiCon:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, repr = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_log = dict()
            train_log['loss'] = []
            train_log['Multi_loss'] = []
            train_log['XiCon_loss'] = []
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                B, T, C = batch_x.shape

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if not self.XiCon:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, repr = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #MSE_loss = F.mse_loss(outputs, batch_y, reduction='none')
                Multi_loss = criterion(outputs, batch_y)
                features = F.normalize(repr, dim=-1)  # B, T, C
                global_pos_labels = timeindex.long()
                local_loss, global_loss = self.XiCon_loss(features, global_pos_labels)


                if self.args.model == 'XiCon_PatchMixer':
                    xicon_loss = (local_loss.reshape(B, C, -1).mean(dim=2).mean(dim=1) + global_loss.mean(dim=0))/2.0
                elif self.args.model == 'XiCon':
                    xicon_loss = (local_loss.reshape(B, C, -1).mean(dim=2).mean(dim=1) + global_loss.mean(dim=0))/2.0
                elif self.args.model == 'XiCon_PatchTST':
                    xicon_loss = (local_loss.reshape(B, C, -1).mean(dim=2).mean(dim=1) + global_loss.mean(dim=0))/2.0                                                                
                else:
                    xicon_loss = (local_loss.mean(dim=1) + global_loss)/2.0

                #loss = MSE_loss.mean() + self.args.AutoCon_lambda * xicon_loss.mean()
                loss = Multi_loss.mean() + self.args.AutoCon_lambda * xicon_loss.mean()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                train_log['loss'].append(loss.item())
                train_log['XiCon_loss'].append(xicon_loss.detach().cpu())
                train_log['Multi_loss'].append(Multi_loss.item())

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_log['loss'] = np.average(train_log['loss'])
            train_log['XiCon_loss'] = torch.cat(train_log['XiCon_loss'], dim=0)
            train_log['Multi_loss'] = np.average(train_log['Multi_loss'])

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} "
                  f"Train Loss: {train_log['loss']:.4f} (Forecasting Loss:{train_log['Multi_loss']:.4f} + "
                  f"XiCon Loss:{train_log['XiCon_loss'].mean():.4f} x Lambda({self.args.AutoCon_lambda})), "
                  f"Vali MSE Loss: {vali_loss:.4f} Test MSE Loss: {test_loss:.4f}")
                        
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        infer_time_sum = 0
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                timeindex = timeindex.float().to(self.device)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if not self.XiCon:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, reprs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if not self.XiCon:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]                                                
                    else:          
                        torch.cuda.synchronize()
                        start = time.time()
                        outputs, reprs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        torch.cuda.synchronize()
                        end = time.time()
                        infer_time_sum = infer_time_sum + (end - start)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # dilate_e, shape_e, temporal_e = shape_metric(preds, trues)  # These metrics take a long time to calculate.
        dilate_e, shape_e,temporal_e  = 0.0,  0.0,  0.0

        print(f'mse:{mse}, mae:{mae}, mape:{mape}, mspe:{mspe} dilate:{dilate_e:.7f}, Shapedtw:{shape_e:.7f}, Temporaldtw:{temporal_e:.7f}')
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        if self.args.save:
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return mse, mae, mape, mspe, dilate_e, shape_e, temporal_e
