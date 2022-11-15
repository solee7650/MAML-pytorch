FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04
LABEL	maintainer="solee"
 
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

#optional if you get the nvidia/repos/ubuntu1804 docker error
# RUN rm /etc/apt/sources.list.d/cuda.list && \
#     rm /etc/apt/sources.list.d/nvidia-ml.list  && \
#     apt-key del 7fa2af80 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install Dependencies of anaconda3
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends\
    wget bzip2 curl git libgl1-mesa-glx &&\ 
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

# Install anaconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm -f ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda update -n base -c defaults conda
#  Create new Env for miniconda

COPY environment.yaml .
COPY requirements.txt .
RUN conda env create -f environment.yaml&& \
     \
    echo "conda activate solee" >> ~/.bashrc

