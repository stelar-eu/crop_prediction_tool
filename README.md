# crop_prediction_tool

## Dockerization step


### Create a Temporary Directory for Docker Installation

<pre>
mkdir /your/address/docker-tmp
</pre>

### Build the Docker Container

CLone the repository :

`git clone https://github.com/stelar-eu/crop_prediction_tool.git`

Then `cd crop_prediction_tool`

and run the following commands to build and save the Docker container:

<pre>
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
DOCKER_TMPDIR=/your/address/docker-tmp \
DOCKER_CONFIG=$HOME/.docker docker build -t docker_fin_3dunet . && \
docker save -o /docker_file/address_in/crop_prediction_tool/docker_fin_3dunet.tar docker_fin_3dunet
</pre>



## Data processing 


Download the LAI dataset from [this dataset](https://imisathena.sharepoint.com/sites/STELAR/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSTELAR%2FShared%20Documents%2FWP5%2DPilot%20Testing%20and%20Evaluation%2FTask%205%2E3%20%2D%20Pilot%202%20%2D%20Early%20crop%20growth%20predictions%2Fdatasets%2Fstudy%20site%20france%2FLAI%2030TYQ&viewid=8f67e1b5%2D0da4%2D49f7%2Da0a5%2Dfbdb223d2562). Download `30TYQ_LAI.zip` and extract

Place the files corresponding to the year **2022** in the directory `/dataset/france2/lai_ras/`. 


## Running the tool in the docekr


Load the Docker image and run the crop prediction application

<pre>
cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet
</pre>


##  Preprocessing

To process .RAS files from vista to .npy format(one .npy image is saved per time point)

<pre>
python lai_extraction_from_ras.py --image_length 10002 --image_width 10002
</pre>


Extract Labels from EuroCrops. Download the crop labels for the required country from https://zenodo.org/records/8229128 and extract and move the files to `./dataset/shape_files_from_Eurocrops`. In `./lai_extraction_from_ras.py` enter the eastings and northings of the selected vista bounding box. Then to extract LAI from RAS files, run  

<pre>
python label_extraction_docker.py --eastings_min 704855.0000 --eastings_max 804875.0000 --northings_min 4895125.0000 --northings_max 4995145.0000
</pre>

These corners of the bounding box available from Vista. 


## Running crop predictor to save the output of the whole tile 

Put all the models inside `/checkpoints_f1` directory.

For crop grown in between February and August
<pre>
cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 

python vista_patch_exp0/vista_testing_comp_f1_docker.py --season Feb_Aug
</pre>


For crop grown in between May and August
<pre>
python vista_patch_exp0/vista_testing_comp_f1_docker.py --season May_Aug
</pre>


For crop grown in between June and October
<pre>
python vista_patch_exp0/vista_testing_comp_f1_docker.py --season Jun_Oct
</pre>


For crop grown in between January and August
<pre>
python vista_patch_exp0/vista_testing_comp_f1_docker.py --season Jan_Aug
</pre>



## Comprehensive evaluation of Ensemble performance to get crop-wise IOU, Accuracy, F1 score and confusion matrix 

## Test set preparation to evaluate the ensemble

### For months February to August

<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 33 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 36 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 41 --season Feb_Aug


python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 34 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 37 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 40 --season Feb_Aug

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 34 --crop_2 37 --crop_3 40
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 33 --crop_2 36 --crop_3 41

</pre>


### For months May to August


<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 2 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 15 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 20 --season May_Aug

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 21 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 23 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 28 --season May_Aug

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 2 --crop_2 15 --crop_3 20
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 21 --crop_2 23 --crop_3 28

</pre>

### For months June to October

<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 8 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 30 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 9 --season Jun_Oct

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 18 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 19 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 7 --season Jun_Oct

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 8 --crop_2 9 --crop_3 30
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 7 --crop_2 18 --crop_3 19

</pre>


### For months January to August

<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 4 --season Jan_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 7 --season Jan_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 9 --season Jan_Aug

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 4 --crop_2 7 --crop_3 9

</pre>



### TO evaluate the model and plot and get quantitative results  

<pre>

python vista_patch_exp0/evaluator.py --season Feb_Aug --num_subgroup_samples 300
python vista_patch_exp0/evaluator.py --season May_Aug --num_subgroup_samples 300
python vista_patch_exp0/evaluator.py --season Jun_Oct --num_subgroup_samples 300
python vista_patch_exp0/evaluator.py --season Jan_Aug --num_subgroup_samples 300

python vista_patch_exp0/metrics_and_visuals.py


</pre>