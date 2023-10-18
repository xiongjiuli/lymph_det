"""Script for training the transoar project."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import monai
from trainer_v1 import Trainer
from data.dataloader_v1 import get_loader
from utils.io_v1 import get_config, write_json, get_meta_data, creat_logging
from models.swin_unet3d_fpn_v1 import swinUnet_3D_fpn
from models.res101_v1 import CenterNet
from models.swin_unet3d_v1 import swinUnet_p_3D
from models.swin_unetr_v1 import SwinUNETR
from criterion_v1 import Criterion
import datetime


def match(n, keywords):
    out = False
    for b in keywords:
        if b in n:
            out = True
            break
    return out

def train(config, args, timestamp):

    log_file_path = f'/public_bme/data/xiongjl/lymph_det/logfile/{config["model_name"]}-{timestamp}.log'
    logger = creat_logging(log_file_path)
    print(f'the loger file :{log_file_path} has be created')
    device = config['device']
    # Build necessary components
    train_loader = get_loader(config, 'training')

    if config['overfit']:
        val_loader = get_loader(config, 'training')
    else:
        val_loader = get_loader(config, 'validation')

    if config['model_name'] == 'swin3d':
        x = torch.randn((1, 1, config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]))
        window_size = [i // 32 for i in x.shape[2:]]
        model = swinUnet_p_3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                            window_size=window_size, in_channel=1, num_classes=64
                            )
        
    elif config['model_name'] == 'swin3d_fpn':
        x = torch.randn((1, 1, config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]))
        window_size = [i // 32 for i in x.shape[2:]]
        model = swinUnet_3D_fpn(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                            window_size=window_size, in_channel=1, num_classes=64
                            )

    elif config['model_name'] == 'unetr':
        model = SwinUNETR(img_size=config['patch_size'], in_channels=1, out_channels=7, feature_size=48)

    elif config['model_name'] == 'res101':
        model = CenterNet('resnet101', 1)

    else:
        print(f'model name is wrong! now the model name is {config["model_name"]}')

    model = model.to(device=device)
    criterion = Criterion(config) # .to(device=device)

    optim = torch.optim.Adam(
        model.parameters(), lr=float(config['lr'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config['scheduler_steps'], gamma=config['gamma'])

    # Load checkpoint if applicable
    if config['resume'] is not False:
        checkpoint = torch.load(Path(config['resume']))
        print(f'the path of the checkpoint is {Path(config["resume"])}')
        print('the model have been loaded @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # checkpoint['scheduler_state_dict']['step_size'] = config['lr_drop']

        # Unpack and load content
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        metric_start_val = checkpoint['metric_max_val']
        # model.load_state_dict(checkpoint['model'])
        # optim.load_state_dict(checkpoint['opt'])
        # epoch = 19
        # metric_start_val = 0.
    else:
        epoch = 0
        metric_start_val = 0.

    # Init logging
    path_to_run = Path(os.getcwd()) / 'runs' / config['model_name']
    path_to_run.mkdir(exist_ok=True)

    # Get meta data and write config to run
    config.update(get_meta_data())
    write_json(config, path_to_run / f'{timestamp}_config.json')

    # Build trainer and start training
    trainer = Trainer(
        train_loader, val_loader, model, criterion, optim, scheduler, device, config, 
        path_to_run, epoch, metric_start_val, logger, timestamp
    )
    trainer.run()
        

if __name__ == "__main__":
    
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # Add minimal amount of args (most args should be set in config files)
    parser.add_argument("--config", type=str, default='lymph_nodes_det')
    # parser.add_argument("--resume", type=str, help="Path to checkpoint to use.", default='/public_bme/data/xiongjl/lymph_det/runs/swin3d/1017003600_model_last.pt')
    # parser.add_argument("--resume", type=str, help="Path to checkpoint to use.", default="/public_bme/data/xiongjl/uii/checkpoints/0912_v1_swin_crop160_hmapv4_best-1000.pt")
    # parser.add_argument("--resume", type=str, help="Path to checkpoint to use.", default=None)
    args = parser.parse_args()

    # Get relevant configs
    config = get_config(args.config)

    # To get reproducable results
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    monai.utils.set_determinism(seed=config['seed'])
    random.seed(config['seed'])

    torch.backends.cudnn.benchmark = False  # performance vs. reproducibility
    torch.backends.cudnn.deterministic = True
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d%H%M%S")
    train(config, args, timestamp)
