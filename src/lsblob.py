"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

from .data import compile_data
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map)
from .models import compile_model

import os
from .blobhead import detect_blob
import numpy as np
import time
# from .socket_server import WebSocketServer


def viz_model_preds_no_mlt(version,
                    modelf,
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=1,
                    viz_train=False,

                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    bsz=4,
                    nworkers=10,
                    save_output=False,
                    debug=True,
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = [
            'CAM_FRONT_LEFT',
            'CAM_FRONT', 
            'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 
            'CAM_BACK', 
            'CAM_BACK_RIGHT'
            ]
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']


    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()

    counter = 0
    def loop():
        free_time = time.time()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
                # every 30 seconds blackout image for 20 seconds
               

                
                # imgs[:, [1, 3], :, :, :] = 0
               
                start = time.time()
                out = model(imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device),
                        )
                end = time.time()
                print("BEV FPS: ", 1/(end-start))
                # BEV FPS: 14* 4(batch size) = 56
                out = out.sigmoid().cpu()
                for si in range(imgs.shape[0]):
                    gray = out[si].squeeze(0).numpy()
                    # clip to 0, 255

                    gray = (gray * 255).astype(np.uint8)
                    start = time.time()
                    detect_blob(gray,save_output=True) 
                    end = time.time()
                    # print("FPS: ", 1/(end-start))
                    # min FPS 497.13215597961363
                    # max FPS 1698.0987854251011
                    # avg FPS 773.7254729661064

                    if save_output:
                            plt.clf()
                            for imgi, img in enumerate(imgs[si]):
                                ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                                showimg = denormalize_img(img)
                                # flip the bottom images
                                if imgi > 2:
                                    showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                                plt.imshow(showimg)
                                plt.axis('off')
                                plt.annotate(
                                    cams[imgi].replace('_', ' '), 
                                    (0.01, 0.92), 
                                    xycoords='axes fraction',
                                    color='white',  # Text color
                                    bbox=dict(
                                        facecolor='black', 
                                        edgecolor='none', 
                                        boxstyle='round,pad=0.5')
                                    )

                            ax = plt.subplot(gs[0, :])
                            ax.get_xaxis().set_ticks([])
                            ax.get_yaxis().set_ticks([])
                            plt.setp(ax.spines.values(), color='b', linewidth=1)
                            plt.legend(handles=[
                                mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                                mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                                mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                            ], loc=(0.01, 0.86))
                            plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
                            # plt.imsave('output_image.png', out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues'
                            # plt.imsave('visualization/bev/output_image.png', out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
                        
                            # plot static map (improves visualization)
                            rec = loader.dataset.ixes[counter]
                            plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                            plt.xlim((out.shape[3], 0))
                            plt.ylim((0, out.shape[3]))
                            add_ego(bx, dx)

                            imname = f'eval{batchi:06}_{si:03}.jpg'
                            path = os.path.join('visualization',imname)
                            print('saving', path)
                            plt.savefig(path)
                            counter += 1
                            free_period += 1
    with torch.no_grad():
        
        while True:
            loop()
