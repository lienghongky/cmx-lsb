"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src
import asyncio

async def detection_loop(version, modelf, dataroot, map_folder):
    await asyncio.to_thread(src.lsblob.viz_model_preds_no_mlt, version, modelf, dataroot, map_folder)
    # src.lsblob.viz_model_preds_no_mlt(
    #             version=version,
    #             modelf=modelf,
    #             dataroot=dataroot,
    #             map_folder=map_folder
    #         )
async def start_server(version, modelf, dataroot, map_folder):
    # Use asyncio.gather() to run the WebSocket server and detection loop concurrently
    websocket_task = asyncio.create_task(src.socket_server.run_websocket_server())  # Create the WebSocket server task
    detection_task = asyncio.create_task(detection_loop(version, modelf, dataroot, map_folder))  # Create detection loop task

    await asyncio.gather(websocket_task, detection_task)  # Wait for both to run concurrently

if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'train': src.train.train,
        'eval_model_iou': src.explore.eval_model_iou,
        'viz_model_preds': src.explore.viz_model_preds,
        'viz_model_preds_no_mlt': src.explore.viz_model_preds_no_mlt,
        'start_server': start_server,
    })