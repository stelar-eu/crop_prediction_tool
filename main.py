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

from vista_patch_exp0.spatial_recon import evaluate_months

from vista_patch_exp0.eval_pre_sampler import LAI_period_sampler

from vista_patch_exp0.subgroup_aggregator import subgroup_aggregator

from vista_patch_exp0.final_evaluator import patch_evaluator 

from vista_patch_exp0.all_metrics_and_visuals import write_eval_results

'''
conda deactivate
conda deactivate
cd /home/luser/crop_prediction_tool
conda activate /home/luser/miniforge3/envs/stcon4

python main.py input.json output.json

'''

download_LAI = False
download_mask = False
start_evaluating = False

pre_sampling_Feb_Aug = False
pre_sampling_May_Aug = False
pre_sampling_Jun_Oct = False
pre_sampling_Jan_Aug = False

Feb_Aug_subgroup_aggregation = False
May_Aug_subgroup_aggregation = False
Jun_Oct_subgroup_aggregation = False
Jan_Aug_subgroup_aggregation = False

do_all_evaluations = False

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
        #mc.put_object(s3_path= outputs['predictions'], file_path="./testpath/evaluation_report.txt")
        #mc.put_object(s3_path= outputs['predictions'], file_path="./evaluation_results/evaluation_report.txt")



        params = json['parameters']
        chosen_months = params['months_chosen']
        ulx = params['upper_left_x']
        uly = params['upper_left_y']
        pixel_size = params['pixel_size']
        crs_code = params['crs_code']

        print("params", params)
        print("which season here", chosen_months)
        

        # Download LAI
        act_lai_files = inputs['actual_LAI']          # path to LAI folder in MinIO
        outputs = json['output']
        if download_LAI:
            for ts in range (len(act_lai_files)):
                mc.get_object(s3_path=act_lai_files[ts], local_path='./dataset/france2/processed_lai_npy2/'+act_lai_files[ts][40:-6]+str(ts).zfill(2)+'.tif')

        # Download Labels
        if download_mask:
            labels = inputs['spatial_labels']          # path to LAI folder in MinIO
            mc.get_object(s3_path=labels[0], local_path='./storage/full_mast1/vista_labes_aligned.npy')

        # To save full tail
        if start_evaluating:
            for period in (range(len(chosen_months))):
                evaluate_months(chosen_months[period], ulx, uly, pixel_size, crs_code)
        



        # for quantitative evaluation
        if pre_sampling_Feb_Aug:
            crop_groups = params['Feb_Aug_crop_group']       
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[0], crop)
        if Feb_Aug_subgroup_aggregation:
            subgroup = params['Feb_Aug_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)
            subgroup = params['Feb_Aug_subgroup2']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)





        if pre_sampling_May_Aug:
            crop_groups = params['May_Aug_crop_group']         
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[1], crop)
        if May_Aug_subgroup_aggregation:
            subgroup = params['May_Aug_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)
            subgroup = params['May_Aug_subgroup2']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)




        if pre_sampling_Jun_Oct:
            crop_groups = params['Jun_Oct_crop_group']       
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[2], crop)
        if Jun_Oct_subgroup_aggregation:
            subgroup = params['Jun_Oct_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)
            subgroup = params['Jun_Oct_subgroup2']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)





        if pre_sampling_Jan_Aug:
            crop_groups = params['Jan_Aug_crop_group']      
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[3], crop)
        if Jan_Aug_subgroup_aggregation:
            subgroup = params['Jan_Aug_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2])


        if do_all_evaluations:
            num_patches = params['num_eval_patches']
            for period in (range(len(chosen_months))):
                patch_evaluator(chosen_months[period], num_patches)

        write_eval_results()

        mc.put_object(s3_path= outputs['predictions'], file_path="./evaluation_results/crop_type_confusion_matrix.png")
        mc.put_object(s3_path= outputs['predictions'], file_path="./evaluation_results/evaluation_report.txt")
        mc.put_object(s3_path= outputs['predictions'], file_path="./evaluation_results/exp2_acc_no_cloud_interpol.png")
        mc.put_object(s3_path= outputs['predictions'], file_path="./evaluation_results/exp2_f1_no_cloud_interpol.png")
        mc.put_object(s3_path= outputs['predictions'], file_path="./evaluation_results/exp2_iou_no_cloud_interpol.png")

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


### "predictions": "s3://vista-bucket/test_directory/random_cards.txt"