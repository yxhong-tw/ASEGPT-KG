import os

from ckiptagger import data_utils

root_dir = os.path.dirname(os.path.abspath(__file__))

data_utils.download_data_gdown(f'{root_dir}/data/ckiptagger/')
