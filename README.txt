
python main.py viz_model_preds mini/trainval --modelf=models/model525000.pt --dataroot=data/nuscenes --map_folder=data/nuscenes/maps

python main.py viz_model_preds mini --modelf=models/model525000.pt --dataroot=datas --map_folder=datas/nuscenes

## Generate Visualization
python main.py viz_model_preds_no_mlt mini --modelf=models/model525000.pt --dataroot=datas --map_folder=datas/mini --save_output=1

## Run server
python main.py start_server  mini --modelf=models/model525000.pt --dataroot=datas --map_folder=datas/mini



## Frames to gif

ffmpeg -framerate 10 -pattern_type glob -i 'eval*.jpg' -vf "scale=962:-1:flags=lanczos,palettegen" -y palette.png & \
ffmpeg -framerate 10 -pattern_type glob -i 'eval*.jpg' -i palette.png -lavfi "scale=962:-1:flags=lanczos [x]; [x][1:v] paletteuse" -y ../imgs/output.gif