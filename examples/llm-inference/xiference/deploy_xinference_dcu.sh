#!/bin/bash
# ==============================================================================
# Xinference on Hygon DCU 自动化部署脚本 v3.0 (Shell 容器模式)
#
# 功能:
# 1. 创建一个交互友好的 Dockerfile (ENTRYPOINT 为 /bin/sh)。
# 2. 构建高度灵活、易于调试的自定义镜像。
# 3. 使用新镜像启动 Xinference 服务。
# ==============================================================================

set -e # 如果任何命令失败，脚本将立即退出

# --- 配置变量 ---
BASE_IMAGE="image.sourcefind.cn:5000/dcu/admin/base/custom:vllm0.8.5-ubuntu22.04-dtk25.04-rc7-das1.5-py3.10-20250514-fixpy-rocblas0513-alpha"
FINAL_IMAGE_NAME="dcu-xinference:1.0"
SERVICE_CONTAINER_NAME="xinference-service"
HOST_PORT="9998"
HOST_CACHE_DIR="$HOME/.cache/huggingface/hub"

# --- 脚本主逻辑 ---
echo "🚀 开始自动化部署 Xinference on DCU (Shell 容器模式)..."

if [ ! -d "$HOST_CACHE_DIR" ]; then
    echo "🟡 宿主机缓存目录 '$HOST_CACHE_DIR' 不存在，正在创建..."
    mkdir -p "$HOST_CACHE_DIR"
    echo "✅ 缓存目录创建成功。"
fi

echo "📝 步骤 1: 生成临时的 Dockerfile (Shell 容器模式)..."
cat <<EOF > Dockerfile.xinference
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
WORKDIR /app
RUN pip install "xinference[vllm]==1.5.1" && \
    pip uninstall -y xoscar && \
    pip install xoscar==0.6.2

# 设置 ENTRYPOINT 为 /bin/sh，提供最佳交互体验
ENTRYPOINT ["/bin/sh"]

# 设置默认执行的命令。使用 'exec' 可以让 xinference-local 继承 shell 的进程号(PID 1)
# 从而能够正确接收 Docker 发出的停止信号，实现优雅停机。
CMD ["-c", "exec xinference-local -H 0.0.0.0 --log-level info"]
EOF
echo "✅ Dockerfile 已生成。"

echo "🛠️ 步骤 2: 构建自定义 Xinference 镜像 '${FINAL_IMAGE_NAME}'..."
docker build -t ${FINAL_IMAGE_NAME} -f Dockerfile.xinference .
echo "✅ 自定义镜像构建成功！"
rm Dockerfile.xinference

if [ "$(docker ps -a -q -f name=^/${SERVICE_CONTAINER_NAME}$)" ]; then
    echo "🟡 发现已存在的服务容器 '${SERVICE_CONTAINER_NAME}'，正在停止并移除..."
    docker stop ${SERVICE_CONTAINER_NAME}
    docker rm ${SERVICE_CONTAINER_NAME}
    echo "✅ 旧容器已移除。"
fi

echo "🚀 步骤 3: 启动 Xinference 服务容器..."
# 在这种模式下，我们可以在 docker run 命令后附加需要执行的命令，来覆盖默认的 CMD
# 这里我们覆盖默认的 CMD，将日志级别设为 debug
docker run \
    -d \
    --name ${SERVICE_CONTAINER_NAME} \
    --restart always \
    -e XINFERENCE_MODEL_SRC=modelscope \
    -p ${HOST_PORT}:9997 \
    --shm-size=16G \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal -v ${HOST_CACHE_DIR}:/home/.cache/huggingface/hub \
    --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    ${FINAL_IMAGE_NAME} \
    -c "exec xinference-local -H 0.0.0.0 --log-level debug"

echo "🎉 恭喜！Xinference 服务已成功启动！"
echo "👉 您可以通过 'docker logs -f ${SERVICE_CONTAINER_NAME}' 查看服务日志。"
echo "👉 您可以通过 'docker exec -it ${SERVICE_CONTAINER_NAME} /bin/bash' 直接进入容器调试。"
echo "👉 Xinference 端点位于: http://<您的服务器IP>:${HOST_PORT}"