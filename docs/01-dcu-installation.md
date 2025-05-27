# DCU 安装配置指南

> 本文档整理自海光DCU开发社区和网络公开资料

## 概述

本指南将帮助您完成海光 DCU (Deep Computing Unit) 加速卡的安装和配置，包括硬件安装、驱动安装、环境部署和验证测试。

## 系统要求

### 支持的操作系统

| 包管理 | 操作系统  | 版本        | 内核                                   | 推荐程度 |
| ------ | --------- | ----------- | -------------------------------------- | -------- |
| rpm    | CentOS    | 7.6         | 3.10.0-957.el7.x86_64                  | ⭐⭐⭐⭐⭐ |
| rpm    | CentOS    | 8.5         | 4.18.0-348.el8.x86_64                  | ⭐⭐⭐⭐   |
| deb    | Ubuntu    | 20.04.1     | 5.4.0-42-generic                       | ⭐⭐⭐⭐⭐ |
| deb    | Ubuntu    | 22.04       | 5.15.0-43-generic                      | ⭐⭐⭐⭐   |
| rpm    | 麒麟      | v10 SP2     | 4.19.90-24.4.v2101.ky10.x86_64         | ⭐⭐⭐    |
| rpm    | UOS       | 1021e       | 4.19.90-2109.1.0.0108.up2.uel20.x86_64 | ⭐⭐⭐    |
| rpm    | openEuler | 22.03       | 5.10.0-60.18.0.50.oe2203.x86_64        | ⭐⭐⭐    |

*注：推荐使用 CentOS 7.6 或 Ubuntu 20.04，内核查看命令：`uname -r`*

### 硬件要求

- 支持 PCIe 3.0/4.0 的主板
- 充足的电源供应（建议 750W 以上）
- 足够的机箱空间和散热条件

## 安装步骤

### 第一步：硬件安装

1. **安装 DCU 加速卡**
   - 关闭计算机电源并断开电源线
   - 将 DCU 加速卡插入主板的 PCIe x16 插槽
   - 连接 PCIe 电源线（如需要）
   - 确保卡片安装牢固

2. **验证硬件识别**
   ```bash
   # 检查 DCU 加速卡是否被系统识别
   lspci | grep -i Display
   ```

### 第二步：驱动安装

#### RPM 系列系统（CentOS、麒麟、UOS、openEuler）

1. **安装依赖包**
   ```bash
   yum install -y \
   cmake \
   automake \
   gcc \
   gcc-c++ \
   rpm-build \
   autoconf \
   kernel-devel-`uname -r` \
   kernel-headers-`uname -r`
   ```

2. **下载并安装驱动**
   ```bash
   # 从开发者社区下载最新驱动
   chmod 755 rock*.run
   ./rock*.run
   reboot
   ```

3. **验证驱动安装**
   ```bash
   lsmod | grep dcu
   ```

#### DEB 系列系统（Ubuntu）

1. **安装依赖包**
   ```bash
   apt update
   apt install -y \
   cmake \
   automake \
   rpm \
   gcc \
   g++ \
   autoconf \
   linux-headers-`uname -r`
   ```

2. **安装驱动**
   ```bash
   chmod 755 rock*.run
   ./rock*.run
   reboot
   ```

3. **验证驱动安装**
   ```bash
   lsmod | grep dcu
   ```

### 第三步：环境部署

#### 方式一：容器化部署（推荐）

1. **安装 Docker**
   ```bash
   # CentOS/RHEL
   yum install -y docker-ce docker-ce-cli containerd.io
   systemctl daemon-reload
   systemctl restart docker
   
   # Ubuntu
   apt-get install -y docker-ce docker-ce-cli containerd.io
   systemctl daemon-reload
   systemctl restart docker
   ```

2. **拉取 DCU 镜像**
   ```bash
   # 从光源镜像仓库拉取 PyTorch 镜像
   docker pull image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.10.0-centos7.6-dtk-22.10-py38-latest
   ```

3. **启动容器**
   ```bash
   docker run \
   -it \
   --name=dcu-dev \
   --device=/dev/kfd \
   --device=/dev/dri \
   --security-opt seccomp=unconfined \
   --cap-add=SYS_PTRACE \
   --ipc=host \
   --network host \
   --shm-size=16G \
   --group-add 39 \
   -v /opt/hyhal:/opt/hyhal \
   image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.10.0-centos7.6-dtk-22.10-py38-latest
   ```

#### 方式二：物理机部署

1. **安装 DTK 依赖**
   ```bash
   # 详细依赖安装请参考完整安装文档
   yum groupinstall -y "Development tools"
   yum install -y epel-release centos-release-scl
   # ... 更多依赖包
   ```

2. **安装 Python 3.8**
   ```bash
   cd /tmp
   wget -O python.tgz https://registry.npmmirror.com/-/binary/python/3.8.12/Python-3.8.12.tgz
   mkdir python-tmp
   tar -xvf python.tgz -C ./python-tmp --strip-components 1
   cd python-tmp
   ./configure --enable-shared
   make -j$(nproc)
   make install
   ```

3. **安装 DTK**
   ```bash
   # 从开发者社区下载 DTK
   tar -xvf DTK-*.tar.gz -C /opt/
   ln -s /opt/dtk-* /opt/dtk
   ```

4. **配置环境变量**
   ```bash
   cat > /etc/profile.d/dtk.sh <<-"EOF"
   #!/bin/bash
   
   # DTK 环境
   source /opt/dtk/env.sh
   
   # Python 环境
   export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/lib64/:$LD_LIBRARY_PATH
   export PATH=/usr/local/bin:$PATH
   
   # 其他工具链环境变量
   # ...
   EOF
   
   source /etc/profile.d/dtk.sh
   ```

### 第四步：环境验证

1. **验证 DCU 环境**
   ```bash
   # 检查 DCU 状态
   rocm-smi
   
   # 检查 DCU 架构信息
   rocminfo | grep gfx
   ```

2. **验证 AI 框架**
   ```bash
   # 验证 PyTorch
   python3 -c "import torch; print('PyTorch version:', torch.__version__); print('DCU available:', torch.cuda.is_available())"
   ```

## 常见问题排查

### 硬件相关

**问题**：`lspci | grep -i Display` 无显示
**解决**：检查 DCU 加速卡是否正确插入 PCIe 插槽，清理金手指

### 驱动相关

**问题**：`lsmod | grep dcu` 无显示
**解决**：
1. 手动加载驱动：`modprobe hydcu`
2. 检查配置文件：`echo "options hydcu hygon_vbios=0" > /etc/modprobe.d/hydcu.conf`
3. 重启系统

**问题**：驱动加载失败
**解决**：检查系统启动参数，删除 `nomodeset` 选项

### 环境相关

**问题**：`rocm-smi` 命令找不到
**解决**：确保正确设置了 DTK 环境变量

**问题**：权限错误
**解决**：将用户添加到 39 组：`usermod -a -G 39 $USER`

## 相关资源

- [海光开发者社区](https://developer.hpccube.com/) - 驱动、DTK、框架下载
- [光源镜像仓库](https://sourcefind.cn/) - Docker 镜像获取
- [开发者论坛](https://forum.hpccube.com/) - 技术讨论
- [DCU FAQ](https://developer.hpccube.com/gitbook//dcu_faq/index.html) - 常见问题解答

## 下一步

安装完成后，您可以：
- 查看 [训练指南](training.md) 了解如何进行模型训练
- 查看 [微调指南](fine-tuning.md) 了解如何进行模型微调
- 查看 [推理指南](inference.md) 了解如何部署模型推理
- 查看 [HPC 指南](hpc.md) 了解科学计算应用
