import json
import sys
import traceback
from utils.mclient import MinioClient
'''from utils.processing import prepare_data
from utils.inference import run_inference
from utils.io_utils import save_geotiff, save_png
from utils.config import transform, crs_code, get_labels_in_color'''
import os
import numpy as np
import string
import random

from vista_patch_exp0.spatial_recon import evaluate_season

'''
conda deactivate
conda deactivate
cd /home/luser/crop_prediction_tool
conda activate /home/luser/miniforge3/envs/stcon4

python main.py input.json output.json

'''

download_LAI = False
download_mask = False



def create_random_txt(filename: str) -> None:

    pool = string.ascii_letters + string.digits + string.punctuation + string.whitespace
    random_text = ''.join(random.choices(pool, k=1000))
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(random_text)

def run(json):
    try:
        ######################## MINIO INIT ########################
        minio_id = json['minio']['id']
        minio_key = json['minio']['key']
        minio_skey = json['minio']['skey']
        minio_endpoint = json['minio']['endpoint_url']

        mc = MinioClient(minio_endpoint, minio_id, minio_key, secure=True, session_token=minio_skey)
        inputs = json['input']
        outputs = json['output']

        create_random_txt("./testpath/random.txt")
        mc.put_object(s3_path= outputs['predictions'], file_path="./testpath/random.txt")

        params = json['parameters']
        which_season = params['month']

        print("params", params)
        print("which season here", which_season)


        # Download LAI
        act_lai_files = inputs['actual_LAI']          # path to LAI folder in MinIO
        outputs = json['output']
        if download_LAI:
            for ts in range (len(act_lai_files)):
                mc.get_object(s3_path=act_lai_files[ts], local_path='./dataset/france2/processed_lai_npy2/'+act_lai_files[ts][40:-6]+str(ts).zfill(2)+'.npy')

        # Download Labels
        if download_mask:
            labels = inputs['spatial_labels']          # path to LAI folder in MinIO
            mc.get_object(s3_path=labels[0], local_path='./storage/full_mast1/vista_labes_aligned.npy')

        evaluate_season()

        return None
    
    except Exception as e:
        print(traceback.format_exc())
        return {
            'message': 'An error occurred during data processing.',
            'error': traceback.format_exc(),
            'status': 500
        }
        

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))
