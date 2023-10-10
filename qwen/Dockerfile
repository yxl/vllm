FROM nvcr.io/nvidia/pytorch:22.12-py3

# 支持的GPU硬件类型 https://developer.nvidia.com/cuda-gpus#compute
# V100 - 7.0, A100 - 8.0, A10/3090 - 8.6
ENV TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6;9.0"
# 检测显卡
#RUN nvidia-smi -L || { echo "未检测到显卡,如果显卡正确配置,使用DOCKER_BUILDKIT=0重新build"; exit 1; }

# 设置为中国国内源
RUN sed -i "s@\(archive\|security\).ubuntu.com@mirrors.aliyun.com@g" /etc/apt/sources.list
RUN echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl-dev \
        ca-certificates \
        make \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# pip 国内加速
RUN pip config set global.index-url https://mirrors.tencentyun.com/pypi/simple

#RUN pip uninstall torch -y --no-cache-dir

# 其中qwen模型的依赖为 auto-gptq tiktoken transformers_stream_generator
RUN pip install einops optimum auto-gptq tiktoken transformers_stream_generator --no-cache-dir

WORKDIR /usr/src

RUN pip install einops optimum --no-cache-dir
COPY test.py .
