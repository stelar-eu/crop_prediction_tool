# Dockerfile, Image, Container
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bash \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_ENV=stconf
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Create conda environment and install packages using 'conda activate' in a single shell
RUN bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda create -y -n $CONDA_ENV python=3.11 && \
    conda activate $CONDA_ENV && \
    conda install -y -c conda-forge cudatoolkit=11.8 && \
    pip install nvidia-cudnn-cu11==8.6.0.163 && \
    conda install -y -c conda-forge tensorflow=2.13 && \
    pip install classification-models-3D==1.0.10 && \
    pip install efficientnet-3D==1.0.2 && \
    pip install segmentation-models-3D==1.0.7 && \
    conda install -y -c conda-forge scikit-learn==1.5.0 && \
    conda install -y -c conda-forge matplotlib && \
    pip install patchify==0.2.3 && \
    conda install -y -c conda-forge scikit-image==0.24.0 && \ 
    conda install -y -c conda-forge rasterio sqlite gdal pyproj fiona && \
    conda install conda-forge::geopandas && \
    conda install conda-forge::opencv && \
    conda clean --all --yes "

# Set working directory
WORKDIR /app

# Copy the entrypoint script into the container
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (can be overridden at runtime)
CMD ["bash"]