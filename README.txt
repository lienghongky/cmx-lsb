
python main.py viz_model_preds mini/trainval --modelf=models/model525000.pt --dataroot=data/nuscenes --map_folder=data/nuscenes/maps

python main.py viz_model_preds mini --modelf=models/model525000.pt --dataroot=datas --map_folder=datas/nuscenes


python main.py viz_model_preds_no_mlt mini --modelf=models/model525000.pt --dataroot=datas --map_folder=datas/mini


python main.py start_server  mini --modelf=models/model525000.pt --dataroot=datas --map_folder=datas/mini