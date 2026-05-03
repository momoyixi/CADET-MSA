"""
Training script for CADET
"""
from run import CADET_run

CADET_run(model_name='CADET', dataset_name='mosi', is_tune=False, seeds=[11], 
# model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_training=True)
