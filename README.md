# crop_prediction_tool

#### Dockerization step


Make a temporary directory for docker installation directory

<pre>
mkdir /your/address/docker-tmp
</pre>

Run the below commands to build the docker container
<pre>
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
DOCKER_TMPDIR=/your/address/docker-tmp \
DOCKER_CONFIG=$HOME/.docker docker build -t docker_fin_3dunet . && \
docker save -o /docker_file/address_in/crop_prediction_tool/docker_fin_3dunet.tar docker_fin_3dunet
</pre>