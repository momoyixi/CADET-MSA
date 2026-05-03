"""
Testing script for CADET
"""
from run import CADET_run

CADET_run(model_name='CADET', dataset_name='mosei', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='test', is_training=False)
