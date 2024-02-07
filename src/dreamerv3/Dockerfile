############  OLD (GPU-based)  ############

# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f dreamerv3/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh scripts/xvfb_run.sh python3 dreamerv3/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir


######MOD######  NEW (CPU-based)  ######MOD######

# 1. Test setup:
# docker run -it --rm ubuntu:20.04 bash

# 2. Start training:
# docker build --platform linux/amd64 -f dreamerv3/Dockerfile -t dreamerv3-img . && \
# docker run -it --platform linux/amd64 --rm --name DreamerV3 \
#   -v ~/logdir:/logdir dreamerv3-img \
#   sh /scripts/xvfb_run.sh python3 dreamerv3/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# Notes:
# - In Macs with M2 chips, need to use '--platform linux/amd64': https://reddit.com/r/docker/comments/155izm4/comment/jsw47ld
#   - This doesn't apply to Linux servers though (like OVH).
# - The 'dreamerv3-img' is the name given to the built Docker image.
# - 'sh /scripts/xvfb_run.sh' is used to run the training script with a virtual framebuffer (Xvfb).
# - Adjust '--configs' and '--task' as needed for your specific training scenario.
# - Mount the local ~/logdir to /logdir in the container to persist training logs.

# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lisbon
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev \
  x11-xserver-utils xvfb \
  && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data
COPY dreamerv3/embodied/scripts scripts
# RUN sh scripts/install-dmlab.sh # needs external `bazel` lib installation
RUN sh scripts/install-atari.sh
# RUN sh scripts/install-minecraft.sh # errored out, prob need external lib too
ENV NUMBA_CACHE_DIR=/tmp
RUN pip3 install crafter
RUN pip3 install dm_control
RUN pip3 install robodesk
RUN pip3 install bsuite

# Agent
RUN pip3 install "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install jaxlib
RUN pip3 install tensorflow_probability
RUN pip3 install optax
# Ensure you're using TensorFlow compatible with the CPU
RUN pip3 install tensorflow
RUN pip3 install tensorboard
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Google Cloud DNS cache (optional)
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600

# Embodied
RUN pip3 install numpy cloudpickle ruamel.yaml rich zmq msgpack
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied

WORKDIR /embodied
