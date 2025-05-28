# 海光DCU加速卡实战环境Docker镜像
# 基于官方DCU基础镜像构建

FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.1.0-centos7.9-dtk-25.04-py310

LABEL maintainer="DCU Community <community@sourcefind.cn>"
LABEL description="海光DCU加速卡实战环境 - 大模型训练、微调、推理与HPC科学计算"
LABEL version="1.0.0"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 设置工作目录
WORKDIR /workspace

# 更新系统并安装必要的包
USER root
RUN yum update -y && \
    yum install -y \
        git \
        wget \
        curl \
        vim \
        htop \
        tmux \
        tree \
        unzip \
        gcc \
        gcc-c++ \
        make \
        cmake \
        openssl-devel \
        libffi-devel \
        zlib-devel \
        bzip2-devel \
        readline-devel \
        sqlite-devel && \
    yum clean all

# 升级pip并安装基础Python包
RUN pip install --upgrade pip setuptools wheel

# 复制requirements文件并安装Python依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 安装可选的高性能推理框架
RUN pip install vllm==0.2.* -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn || echo "vLLM安装失败，跳过"

# 安装PEFT和量化库
RUN pip install peft>=0.6.0 bitsandbytes>=0.41.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 安装Jupyter Lab和扩展
RUN pip install jupyterlab ipywidgets jupyter_contrib_nbextensions \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn && \
    jupyter contrib nbextension install --system

# 创建用户
RUN useradd -m -s /bin/bash dcu && \
    echo "dcu:dcu" | chpasswd && \
    usermod -aG wheel dcu

# 复制项目文件
COPY . /workspace/dcu-in-action/
RUN chown -R dcu:dcu /workspace

# 切换到普通用户
USER dcu

# 设置Jupyter配置
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# 设置环境变量
ENV PYTHONPATH=/workspace/dcu-in-action:$PYTHONPATH
ENV PATH=/workspace/dcu-in-action/scripts:$PATH

# 暴露端口
EXPOSE 8888 8000 7860

# 设置启动脚本
COPY docker-entrypoint.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
USER dcu

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('DCU可用:', torch.cuda.is_available())" || exit 1

# 默认命令
CMD ["/usr/local/bin/docker-entrypoint.sh"] 