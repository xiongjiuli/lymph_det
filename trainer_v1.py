"""Module containing the trainer of the transoar project."""

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluator_v1 import DetectionEvaluator
# from transoar.inference import inference
from plot.sample_2 import plot

class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, config, path_to_run, epoch, metric_start_val
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
        self._scaler = GradScaler()

        self._evaluator = DetectionEvaluator(config)
        self._main_metric_max_val = metric_start_val

    def _train_one_epoch(self, num_epoch):
        self._model.train()
        # self._criterion.train()

        loss_agg = 0
        loss_hmap_agg = 0
        loss_whd_agg = 0
        loss_offset_agg = 0
        for dct in tqdm(self._train_loader):
            # Put data to gpu
            image = dct['input'].to(device=self._device)
            hmap_target = dct['hmap'].to(device=self._device)
            whd_target = dct['whd'].to(device=self._device)
            offset_target = dct['offset'].to(device=self._device)
            mask_target = dct['mask'].to(device=self._device)
            
            # Make prediction
            with autocast(): # 这个是混合精度 autocast + GradScaler
                hmap_pred, whd_pred, offset_pred = self._model(image)
                loss_dict = self._criterion(hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target)

                # Create absolute loss and mult with loss coefficient
                loss_abs = 0
                for loss_key, loss_val in loss_dict.items():
                    loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            self._optimizer.zero_grad()
            self._scaler.scale(loss_abs).backward()

            # Clip grads to counter exploding grads
            max_norm = self._config['clip_max_norm']
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm) # 这个是为了剪裁梯度的，防止梯度爆炸

            self._scaler.step(self._optimizer)
            self._scaler.update()

            loss_agg += loss_abs.item()
            loss_hmap_agg += loss_dict['hmap'].item()
            loss_whd_agg += loss_dict['whd'].item()
            loss_offset_agg += loss_dict['offset'].item()

        loss = loss_agg / len(self._train_loader)
        loss_hmap = loss_hmap_agg / len(self._train_loader)
        loss_whd = loss_whd_agg / len(self._train_loader)
        loss_offset = loss_offset_agg / len(self._train_loader)

        self._write_to_logger(
            num_epoch, 'train', 
            total_loss=loss,
            hmap_loss=loss_hmap,
            whd_loss=loss_whd,
            offset_loss=loss_offset,
        )

    @torch.no_grad()
    def _validate(self, num_epoch):
        self._model.eval()
        # self._criterion.eval()
        loss_agg = 0
        loss_hmap_agg = 0
        loss_whd_agg = 0
        loss_offset_agg = 0
        for dct in tqdm(self._val_loader):
            # Put data to gpu
            image = dct['input'].to(device=self._device)
            hmap_target = dct['hmap'].to(device=self._device)
            whd_target = dct['whd'].to(device=self._device)
            offset_target = dct['offset'].to(device=self._device)
            mask_target = dct['mask'].to(device=self._device)

            # Make prediction
            with autocast():
                hmap_pred, whd_pred, offset_pred = self._model(image)
                loss_dict = self._criterion(hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target)

                # Create absolute loss and mult with loss coefficient
                loss_abs = 0
                for loss_key, loss_val in loss_dict.items():
                    loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            loss_agg += loss_abs.item()
            loss_hmap_agg += loss_dict['hmap'].item()
            loss_whd_agg += loss_dict['whd'].item()
            loss_offset_agg += loss_dict['offset'].item()

        loss = loss_agg / len(self._val_loader)
        loss_hmap = loss_hmap_agg / len(self._val_loader)
        loss_whd = loss_whd_agg / len(self._val_loader)
        loss_offset = loss_offset_agg / len(self._val_loader)

        if loss < self._config['best_loss']:
            txt_paths = self._evaluator(self._model, num_epoch) # generate the txt file
            metric_scores = plot(self._config, txt_paths, num_epoch)
            if metric_scores['AP_IoU_0.01'] >= self._main_metric_max_val \
                and not self._config['debug_mode']:
                self._main_metric_max_val = metric_scores['AP_IoU_0.01']
                self._save_checkpoint(
                    num_epoch,
                    f'model_best_{self._config["bset_ap"]}_{num_epoch}.pt'
                )

        # metric_scores = self._evaluator.eval()

        # Write to logger
        self._write_to_logger(
            num_epoch, 'val', 
            total_loss=loss,
            hmap_loss=loss_hmap,
            whd_loss=loss_whd,
            offset_loss=loss_offset,
        )

        self._write_to_logger(
            num_epoch, 'val_metric',
            AP01=metric_scores['AP_IoU_0.01'],
            AP10=metric_scores['AP_IoU_0.10'],
            AP50=metric_scores['AP_IoU_0.50'],
            AP75=metric_scores['AP_IoU_0.75'],
        )

    def run(self):
        if self._epoch_to_start == 0:   # For initial performance estimation
            self._validate(0)

        for epoch in range(self._epoch_to_start + 1, self._config['epochs'] + 1):
            self._train_one_epoch(epoch)

            # Log learning rates
            self._write_to_logger(
                epoch, 'lr',
                backbone=self._optimizer.param_groups[0]['lr'],
                neck=self._optimizer.param_groups[1]['lr']
            )

            if epoch % self._config['val_interval'] == 0:
                self._validate(epoch)

            self._scheduler.step()

            if not self._config['debug_mode']:
                self._save_checkpoint(epoch, 'model_last.pt')

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
            'metric_max_val': self._config["bset_ap"],
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
        }, self._path_to_run / name)
