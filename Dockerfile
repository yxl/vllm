FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev

# 设置为中国国内源
RUN sed -i "s@\(archive\|security\).ubuntu.com@mirrors.aliyun.com@g" /etc/apt/sources.list
RUN echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace

# pip 国内加速
RUN pip config set global.index-url https://mirrors.tencentyun.com/pypi/simple

# install build and runtime dependencies
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --default-timeout=120

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt --default-timeout=120

# image to build pytorch extensions
FROM dev AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt --default-timeout=120

# copy input files
COPY csrc csrc
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY vllm/__init__.py vllm/__init__.py

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# max jobs used by Ninja to build extensions
ARG max_jobs=10
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=40
ENV NVCC_THREADS=$nvcc_threads

RUN python3 setup.py build_ext --inplace

# image to run unit testing suite
FROM dev AS test

# copy pytorch extensions separately to avoid having to rebuild
# when python code changes
COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY tests tests
COPY vllm vllm

ENTRYPOINT ["python3", "-m", "pytest", "tests"]

# use CUDA base as CUDA runtime dependencies are already installed via pip
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS vllm-base

# 设置为中国国内源
RUN sed -i "s@\(archive\|security\).ubuntu.com@mirrors.aliyun.com@g" /etc/apt/sources.list
RUN echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

# libnccl required for ray
RUN apt-get update -y \
    && apt-get install -y python3-pip

# pip 国内加速
RUN pip config set global.index-url https://mirrors.tencentyun.com/pypi/simple

WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --default-timeout=120

FROM vllm-base AS vllm
COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY vllm vllm

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.api_server"]

# openai api server alternative
FROM vllm-base AS vllm-openai
# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate tiktoken --default-timeout=120

COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY vllm vllm

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
