"""Module containing the trainer of the transoar project."""

import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from evaluator_v1 import DetectionEvaluator
# from transoar.inference import inference
from plot.sample_2 import plot
from utils.io_v1 import npy2nii

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)



class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, config, path_to_run, epoch, metric_start_val, logger, timestamp
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._path_to_run = path_to_run
        self._epoch_to_start = epoch
        self._config = config
        self._writer = SummaryWriter(log_dir=path_to_run)
        # self._scaler = GradScaler()

        self._evaluator = DetectionEvaluator(config)
        self._main_metric_max_val = metric_start_val
        self._logger = logger
        self._timestamp = timestamp

    def _train_one_epoch(self, num_epoch):
        print(f'> start training ...{num_epoch}')
        self._model.train()
        # self._criterion.train()

        loss_agg = 0
        loss_hmap_agg = 0
        loss_whd_agg = 0
        loss_offset_agg = 0
        # for dct in tqdm(self._train_loader):
        pbar = tqdm(self._train_loader)
        for i, dct in enumerate(pbar):
            # Put data to gpu
            image = dct['input'].to(device=self._device)
            hmap_target = dct['hmap'].to(device=self._device)
            whd_target = dct['whd'].to(device=self._device)
            offset_target = dct['offset'].to(device=self._device)
            mask_target = dct['mask'].to(device=self._device)
            if self._config['save_for_see'] and (num_epoch%10)==0:
                npy2nii(image, f'train_image_{self._timestamp}')
                npy2nii(hmap_target, f'train_hmap_target_{self._timestamp}')
                npy2nii(whd_target, f'train_whd_target_{self._timestamp}')
                npy2nii(offset_target, f'train_offset_target_{self._timestamp}')
                npy2nii(mask_target, f'train_mask_target_{self._timestamp}')
            # Make prediction
            # with autocast(): # 这个是混合精度 autocast + GradScaler
            hmap_pred, whd_pred, offset_pred = self._model(image)

            loss_dict = self._criterion(hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target)
            if self._config['save_for_see'] and (num_epoch%10)==0:
                npy2nii(image, f'imagepatch_intrain_{self._timestamp}')
                npy2nii(hmap_pred, f'pred_hmappatch_intrain_{self._timestamp}')

            # Create absolute loss and mult with loss coefficient
            loss_abs = 0
            for loss_key, loss_val in loss_dict.items():
                loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            self._optimizer.zero_grad()
            loss_abs.backward()
            self._optimizer.step()
            # self._scaler.scale(loss_abs).backward()
            
            # Clip grads to counter exploding grads
            # max_norm = self._config['clip_max_norm']
            # if max_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm) # 这个是为了剪裁梯度的，防止梯度爆炸

            # self._scaler.step(self._optimizer)
            # self._scaler.update()
            # loss_abs.backward()

        loss_agg += loss_abs.item()
        loss_hmap_agg += loss_dict['hmap'].item()
        loss_whd_agg += loss_dict['whd'].item()
        loss_offset_agg += loss_dict['offset'].item()
        pbar.set_description(f"Training ")

        loss = loss_agg / len(self._train_loader)
        loss_hmap = loss_hmap_agg / len(self._train_loader)
        loss_whd = loss_whd_agg / len(self._train_loader)
        loss_offset = loss_offset_agg / len(self._train_loader)

        self._logger.info('Epoch: %d, train_Loss: %.4f', num_epoch, loss)
        self._logger.info('Epoch: %d, hmap_Loss: %.4f', num_epoch, loss_hmap)
        self._logger.info('Epoch: %d, whd_Loss: %.4f', num_epoch, loss_whd)
        self._logger.info('Epoch: %d, r_Loss: %.4f', num_epoch, loss_offset)
        print('logger info have done (training)')
        self._write_to_logger(
            num_epoch, 'train', 
            total_loss=loss,
            hmap_loss=loss_hmap,
            whd_loss=loss_whd,
            offset_loss=loss_offset,
        )


    @torch.no_grad()
    def _validate(self, num_epoch):
        print(f'> start validation ... {num_epoch}')
        self._model.eval()
        # self._criterion.eval()
        loss_agg = 0
        loss_hmap_agg = 0
        loss_whd_agg = 0
        loss_offset_agg = 0
        # for dct in tqdm(self._val_loader):
        pbar = tqdm(self._val_loader)
        for i, dct in enumerate(pbar):
            # Put data to gpu
            image = dct['input'].to(device=self._device)
            hmap_target = dct['hmap'].to(device=self._device)
            whd_target = dct['whd'].to(device=self._device)
            offset_target = dct['offset'].to(device=self._device)
            mask_target = dct['mask'].to(device=self._device)
            if self._config['save_for_see'] and (num_epoch%10)==0:
                npy2nii(image, f'valid_image_{self._timestamp}')
                npy2nii(hmap_target, f'valid_hmap_target_{self._timestamp}')
                npy2nii(whd_target, f'valid_whd_target_{self._timestamp}')
                npy2nii(offset_target, f'valid_offset_target_{self._timestamp}')
                npy2nii(mask_target, f'valid_mask_target_{self._timestamp}')
            # Make prediction
            # with autocast():
            hmap_pred, whd_pred, offset_pred = self._model(image)

            loss_dict = self._criterion(hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target)
            if self._config['save_for_see'] and (num_epoch%10)==0:
                npy2nii(image, f'imagepatch_invalid_{self._timestamp}')
                npy2nii(hmap_pred, f'pred_hmappatch_invalid_{self._timestamp}')
            # Create absolute loss and mult with loss coefficient
            loss_abs = 0
            for loss_key, loss_val in loss_dict.items():
                loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            loss_agg += loss_abs.item()
            loss_hmap_agg += loss_dict['hmap'].item()
            loss_whd_agg += loss_dict['whd'].item()
            loss_offset_agg += loss_dict['offset'].item()
            pbar.set_description(f"Validation ")

        loss = loss_agg / len(self._val_loader)
        loss_hmap = loss_hmap_agg / len(self._val_loader)
        loss_whd = loss_whd_agg / len(self._val_loader)
        loss_offset = loss_offset_agg / len(self._val_loader)

        self._logger.info('Epoch: %d, valid_Loss: %.4f', num_epoch, loss)
        self._logger.info('Epoch: %d, valid_hmap_Loss: %.4f', num_epoch, loss_hmap)
        self._logger.info('Epoch: %d, valid_whd_Loss: %.4f', num_epoch, loss_whd)
        self._logger.info('Epoch: %d, valid_r_Loss: %.4f', num_epoch, loss_offset)
        print('logger info have done (validation)')

        if loss < self._config['best_loss']:
            txt_paths = self._evaluator(self._model, num_epoch, self._timestamp) # generate the txt file
            if len(txt_paths) == 0:   # txt_path is the whole path 
                print('txt_path is None...mean no nodes be detected, so will not do the plot function, and no image and txt will be saved')
            else:
                metric_scores = plot(self._config, txt_paths, num_epoch, self._timestamp)
                print(metric_scores['AP_IoU_0.01'])
                if metric_scores['AP_IoU_0.01'] >= self._main_metric_max_val: 
                    # and not self._config['debug_mode']:
                    self._main_metric_max_val = metric_scores['AP_IoU_0.01']
                    self._save_checkpoint(
                        num_epoch,
                        f'{self._timestamp}_model_best_{self._main_metric_max_val}_{num_epoch}.pt'
                    )

                self._logger.info('Epoch: %d, AP_IoU_0.01: %.4d', num_epoch, metric_scores["AP_IoU_0.01"])
                # self._logger.info('Epoch: %d, AP_IoU_0.10: %.4d', num_epoch, metric_scores["AP_IoU_0.10"])
                self._logger.info('Epoch: %d, AP_IoU_0.50: %.4d', num_epoch, metric_scores["AP_IoU_0.50"])
                # self._logger.info('Epoch: %d, AP_IoU_0.75: %.4d', num_epoch, metric_scores["AP_IoU_0.75"])
                self._logger.info('Epoch: %d, AP_IoU_0.01_train: %.4d', num_epoch, metric_scores["AP_IoU_0.01_train"])
                # self._logger.info('Epoch: %d, AP_IoU_0.10_train: %.4d', num_epoch, metric_scores["AP_IoU_0.10_train"])
                self._logger.info('Epoch: %d, AP_IoU_0.50_train: %.4d', num_epoch, metric_scores["AP_IoU_0.50_train"])
                # self._logger.info('Epoch: %d, AP_IoU_0.75_train: %.4d', num_epoch, metric_scores["AP_IoU_0.75_train"])


                self._write_to_logger(
                    num_epoch, 'val_metric',
                    AP01=metric_scores['AP_IoU_0.01'],
                    # AP10=metric_scores['AP_IoU_0.10'],
                    AP50=metric_scores['AP_IoU_0.50'],
                    # AP75=metric_scores['AP_IoU_0.75'],
                )

        # Write to logger
        self._write_to_logger(
            num_epoch, 'val', 
            total_loss=loss,
            hmap_loss=loss_hmap,
            whd_loss=loss_whd,
            offset_loss=loss_offset,
        )
        

    def run(self):
        print(f'start to run the training, and the self._epoch_to_start is {self._epoch_to_start}')
        if self._epoch_to_start == 0:   # For initial performance estimation
            self._validate(0)

        for epoch in range(self._epoch_to_start + 1, self._config['epochs'] + 1):
            self._train_one_epoch(epoch)

            # # Log learning rates
            # self._write_to_logger(
            #     epoch, 'lr',
            #     backbone=self._optimizer.param_groups[0]['lr'],
            #     # neck=self._optimizer.param_groups[1]['lr']
            # )

            if epoch % self._config['val_interval'] == 0:
                self._validate(epoch)

            self._scheduler.step()

            # if not self._config['debug_mode']:
            self._save_checkpoint(epoch, f'{self._timestamp}_model_last.pt')

    def _write_to_logger(self, num_epoch, category, **kwargs):
        for key, value in kwargs.items():
            name = category + '/' + key
            self._writer.add_scalar(name, value, num_epoch)

    def _save_checkpoint(self, num_epoch, name):
        # Delete prior best checkpoint
        if 'best' in name:
            [path.unlink() for path in self._path_to_run.iterdir() if 'best' in str(path)]

        torch.save({
            'epoch': num_epoch,
            'metric_max_val': self._main_metric_max_val,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
        }, self._path_to_run / name)
