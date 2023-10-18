import numpy as np
import torch
from models.swin_unet3d_fpn_v1 import swinUnet_3D_fpn
from models.res101_v1 import CenterNet
from models.swin_unet3d_v1 import swinUnet_p_3D
from models.swin_unetr_v1 import SwinUNETR
from tqdm import tqdm
from evaluator_v1 import *
from data.preprocessor import generate_data
from utils.io_v1 import npy2nii


def stage_one(config, name, part):
    device = config['device']
    data_root_path = Path(config['lymph_nodes_data_path'])
    cascade_path = Path(f'{data_root_path}/{part}_npy/{name}_{config["model_name"]}_cascade.npy')
    if cascade_path.exists():
        result_hmap = np.load(cascade_path)
    else:
        if config['model_name'] == 'swin3d':
            x = torch.randn((1, 1, config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]))
            window_size = [i // 32 for i in x.shape[2:]]
            model = swinUnet_p_3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                                window_size=window_size, in_channel=1, num_classes=64)
            
        elif config['model_name'] == 'swin3d_fpn':
            x = torch.randn((1, 1, config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]))
            window_size = [i // 32 for i in x.shape[2:]]
            model = swinUnet_3D_fpn(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                                window_size=window_size, in_channel=1, num_classes=64)

        elif config['model_name'] == 'unetr':
            model = SwinUNETR(img_size=config['patch_size'], in_channels=1, out_channels=7, feature_size=48)

        elif config['model_name'] == 'res101':
            model = CenterNet('resnet101', 1)

        else:
            print(f'model name is wrong! now the model name is {config["model_name"]}')

        model = model.to(device=device)
        # model = torch.nn.parallel.DataParallel(model)
        model_path = config['cascade']['stage_one_model_path']
        model.load_state_dict(torch.load(model_path)['model']) # model_state_dict

        # 冻结模型的所有参数
        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        step = [pahs - ovlap for pahs, ovlap in zip(config['patch_size'], config['overlap'])]
        
        image_path = data_root_path.joinpath(f'{part}_npy').joinpath(f'{name}_image.npy')
        hmap_path = data_root_path.joinpath(f'{part}_npy').joinpath(f'{name}_hmap.npy')
        if hmap_path.exists() and image_path.exists():
            hmap_target = np.load(hmap_path)
            image_data = np.load(image_path)
        else:
            image_data, hmap_target = generate_data(data_root_path, part, name)
        image_patches, arr_pad_shape = sliding_window_3d_volume_padded(image_data, patch_size=config['patch_size'], stride=step)
        whole_hmap = np.zeros(arr_pad_shape)
        for image_patch in image_patches:
            with torch.no_grad():
                image_input = image_patch['image'].unsqueeze(0)
                point = image_patch['point'][1:]
                image_input = image_input.cuda()
                # print(f'in cascade, the shape of the image_input is {image_input.shape}')
                pred_hmap, _, _ = model(image_input)
                # print('save nii in cascade!!!!')
                # npy2nii(image_input, f'incascade_imagepatch_{point}')
                # npy2nii(pred_hmap, f'incascade_pred_hmappatch_{point}')
                whole_hmap = place_small_image_in_large_image(whole_hmap, pred_hmap.squeeze(0).squeeze(0).cpu(), point)

        whole_hmap = whole_hmap[0:image_data.shape[-3], 0:image_data.shape[-2], 0:image_data.shape[-1]]
        result_hmap = np.abs(whole_hmap - hmap_target)
        np.save(cascade_path, result_hmap)

    non_zero_positions = np.argwhere(result_hmap >= 0.5)
    center = random.choice(non_zero_positions)

    return center
