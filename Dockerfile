# 基于海光DCU官方镜像
FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.4.1-ubuntu22.04-dtk25.04-py3.10

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONPATH=/workspace
ENV DCU_VISIBLE_DEVICES=0,1,2,3
ENV HIP_VISIBLE_DEVICES=0,1,2,3
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip setuptools wheel

# 复制requirements文件
COPY requirements.txt /workspace/

# 安装Python依赖
RUN pip install -r requirements.txt

# 安装额外的开发工具
RUN pip install \
    jupyter \
    jupyterlab \
    tensorboard \
    wandb \
    mlflow \
    prometheus-client \
    psutil \
    gpustat \
    py3nvml

# 复制项目文件
COPY . /workspace/

# 设置权限
RUN chmod +x /workspace/common/setup/*.sh
RUN chmod +x /workspace/scripts/**/*.sh

# 创建必要的目录
RUN mkdir -p /workspace/logs \
    /workspace/data \
    /workspace/models \
    /workspace/checkpoints \
    /workspace/outputs

# 安装项目
RUN pip install -e .

# 暴露端口
EXPOSE 8888 6006 8000 8001

# 启动命令
CMD ["bash"] 