# Image for the RAAS distillation job on Yandex DataSphere (g2.1 = A100 80GB).
#
# Runtime base, not devel, and deliberately so. The only thing devel would buy
# is nvcc for Mask2Former's MSDeformAttn CUDA kernel; without it the pixel
# decoder falls back to a pure-PyTorch implementation at roughly 2x the cost,
# which across this whole project is about nine minutes of A100 time. In
# exchange the image drops from ~15 GB to ~6 GB, so every build and every pull
# is three times cheaper.
#
# detectron2 still compiles, but only its CPU extensions: with no GPU on the
# build host and CUDA_HOME unset, its setup.py picks the CppExtension path,
# which needs g++ alone. Mask2Former's semantic path never calls detectron2's
# CUDA ops (those belong to the R-CNN detectors), so this costs nothing at
# inference.
#
# The build context is a directory holding BOTH checkouts side by side, because
# the image needs the RAAS tree (detectron2, Mask2Former, Maskomaly, vendored
# CLIP and SegFormer) as well as this repository:
#
#   build-context/
#     detectron2/  Mask2Former/  Maskomaly/  vendor/   <- the RAAS checkout
#     Open_Road/                                       <- this repository
#
# Build for linux/amd64 on a cloud VM, not on an Apple-silicon laptop:
#   docker build --platform linux/amd64 \
#       -f Open_Road/methods/raas_distill/train/Dockerfile -t raas-distill .
#
# Parameterised because Docker Hub is not reachable from every network this
# builds on; the build script falls back to cr.yandex/mirror when it isn't.
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

# archive.ubuntu.com is not reachable from this cloud, while the host VM's own
# mirror is. The build script passes that mirror in rather than hardcoding one.
ARG APT_MIRROR=""
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g; \
                s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g; \
                s|http://ports.ubuntu.com/ubuntu-ports|${APT_MIRROR}|g" \
            /etc/apt/sources.list; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        build-essential ninja-build git ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# torch 2.1.2 is the newest release the vendored detectron2 0.6 builds against
# cleanly; later versions drop APIs it still calls.
# Two pins that are load-bearing, not housekeeping:
#  * numpy<2 -- torch 2.1.2's extensions were compiled against NumPy 1.x and
#    abort at import under 2.x;
#  * setuptools<70 -- setuptools 81 dropped pkg_resources, which detectron2
#    0.6's setup.py still imports, while PEP 660 needs >=64.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "numpy<2" "setuptools<70" wheel && \
    # From PyPI, not download.pytorch.org: that host is unreachable from this
    # cloud about half the time, and torch 2.1.2's default PyPI wheel is
    # already the CUDA 12.1 build, which is what this base image provides.
    python3 -m pip install torch==2.1.2 torchvision==0.16.2

WORKDIR /opt/raas
COPY detectron2 /opt/raas/detectron2
COPY Mask2Former /opt/raas/Mask2Former
COPY Maskomaly /opt/raas/Maskomaly

# Two things this line has to work around:
#  * --no-build-isolation, because detectron2's setup.py imports torch at module
#    level and PEP 517's isolated environment does not have it;
#  * a non-editable install, because the vendored setuptools predates PEP 660
#    and rejects `-e` with "missing the 'build_editable' hook". Nothing edits
#    detectron2 at run time, so editable buys us nothing anyway.
RUN python3 -m pip install --no-build-isolation /opt/raas/detectron2

# github.com and huggingface.co are not reachable from inside this build, so
# vm-build.sh fetches them on the host and they arrive through the context.
COPY vendor/road-anomaly-benchmark /opt/raas/third_party/road-anomaly-benchmark
COPY vendor/CLIP /opt/raas/vendor/CLIP
COPY vendor/segformer-b0 /opt/raas/vendor/segformer-b0
# clip.load('ViT-B/32') downloads from openaipublic.azureedge.net on first
# use; seeding its cache keeps the teacher offline at run time too.
COPY vendor/clip-cache /root/.cache/clip

RUN python3 -m pip install \
        "numpy<2" cython scipy shapely timm h5py scikit-image easydict \
        opencv-python-headless ftfy regex tqdm \
        transformers==4.44.2 onnx==1.16.2 onnxruntime==1.19.2 boto3==1.34.162 && \
    python3 -m pip install --no-deps --no-build-isolation /opt/raas/vendor/CLIP

# The student initialises from this local copy, so nothing reaches out to
# huggingface.co at run time either.
ENV STUDENT_BACKBONE=/opt/raas/vendor/segformer-b0
RUN python3 -c "\
from transformers import SegformerForSemanticSegmentation as M; \
m = M.from_pretrained('/opt/raas/vendor/segformer-b0'); \
print('segformer ok:', m.decode_head.classifier.out_channels, 'classes')"

# This repository, checked out beside the RAAS tree rather than inside it. The
# stages no longer derive the RAAS root from their own location -- $RAAS_ROOT
# says where it is -- so Open_Road can live wherever it likes.
COPY Open_Road /opt/open_road
RUN python3 -m pip install --no-deps -e /opt/open_road && \
    python3 -m pip install -r /opt/open_road/methods/raas_distill/train/requirements.txt

ENV RAAS_ROOT=/opt/raas \
    OPEN_ROAD_ROOT=/opt/open_road \
    PYTHONPATH=/opt/raas/Maskomaly/detectron2_replacements:/opt/raas/Maskomaly/maskomaly:/opt/raas/third_party/road-anomaly-benchmark
ENTRYPOINT ["/opt/open_road/methods/raas_distill/train/entrypoint.sh"]
