# DCU安装快速上手步骤

### 系统要求

| 包管理 | 操作系统  | 版本        | 内核                                   | 参考下载地址                                                 |
| ------ | --------- | ----------- | -------------------------------------- | ------------------------------------------------------------ |
| rpm    | CentOS    | 7.6         | 3.10.0-957.el7.x86_64                  | 链接：https://pan.baidu.com/s/1cpb3O_5xmvLcAGtekFhxIg 提取码：q2zq |
| rpm    | CentOS    | 8.5         | 4.18.0-348.el8.x86_64                  |                                                              |
| rpm    | NFS       | 3.2         | 3.10.0-957.nfs.5.x86_64                | 链接：https://pan.baidu.com/s/1YyOMbKlr1BiiGVkRNvbPQw 提取码：8h03 |
| rpm    | NFS       | 4.0         | 4.19.113-11.nfs4.x86_64                |                                                              |
| rpm    | NFS       | 4.0-Desktop | 5.4.0-49-generic                       |                                                              |
| rpm    | Anolis    | 8.4         | 4.19.91-23.4.an8.x86_64                | 链接：https://mirrors.openanolis.cn/anolis/                  |
| rpm    | Anolis    | 8.6         | 4.19.91-26.an8.x86_64                  |                                                              |
| rpm    | Kylin     | v10 SP2     | 4.19.90-24.4.v2101.ky10.x86_64         | 链接：http://distro-images.kylinos.cn:8802/web_pungi/download/share/wA7vpuh4S5ZrxLWRXVBgGO0d9TfJqijD |
| rpm    | UOS       | 1021e       | 4.19.90-2109.1.0.0108.up2.uel20.x86_64 | 链接：https://www.chinauos.com/resource/download-server      |
| rpm    | openEuler | 22.03       | 5.10.0-60.18.0.50.oe2203.x86_64        | 链接：https://openeuler.org/zh/download/                     |
| deb    | Ubuntu    | 20.04.1     | 5.4.0-42-generic                       | 链接：https://pan.baidu.com/s/1CS8nAsOM8zvKIU3eB4WqAg 提取码：ksk3 |
| deb    | Ubuntu    | 22.04       | 5.15.0-43-generic                      |                                                              |

*注：建议使用Centos7.6或者Ubuntu20.04，内核查看命令：`uname -r`*

您仅需以下几步便可以开始体验DCU加速卡：

- 第一步：[硬件安装](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#a)；
- 第二步：[驱动安装](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#b)；
- 第三步：[环境部署](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#c)；
- 第四步：[实例测试](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#d)；

## 第一步：硬件安装

1. 将DCU加速卡插入主板的PCIe插槽，并连接好电源线，如图所示：

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/chaka.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/chaka.png)

2. 通过`lspci | grep -i Display`命令查看是否检测到DCU加速卡，如图所示一张DCU加速卡被识别：（[问题排查1](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#1)）

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/jiance1.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/jiance1.png)

## 第二步：驱动安装

##### ➡rpm系列系统（CentOS，NFS，Anolis，Kylin，UOS，openEuler）

1. **安装驱动依赖包**

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

2. **获取驱动**

   *注：可前往[开发者社区](https://developer.hpccube.com/)→**资源工具**→**驱动**，获取**latest**驱动下载地址。*

   ![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kaifazqudong.png)

3. **安装驱动**

   *注：卸载驱动请执行命令`rpm -e rock`。*

   ```bash
   chmod 755 rock*.run \
   && ./rock*.run \
   && reboot
   ```

4. **验证**（[问题排查2、3](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#2)）

   通过`lsmod | grep dcu`命令验证驱动是否安装成功，如图所示：

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/qudongjiance.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/qudongjiance.png)

##### ➡deb系列系统（Ubuntu）

1. **安装驱动依赖包**

   ```bash
   apt update \
   && apt install -y \
   cmake \
   automake \
   rpm \
   gcc \
   g++ \
   autoconf \
   linux-headers-`uname -r`
   ```

2. **获取驱动**

   *注：可前往[开发者社区](https://developer.hpccube.com/)→**资源工具**→**驱动**，获取**latest**驱动下载地址*

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kaifazqudong.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kaifazqudong.png)

3. **安装驱动**

   *注：卸载驱动请执行命令`apt-get remove rock*`。*

   ```bash
   chmod 755 rock*.run \
   && ./rock*.run \
   && reboot
   ```

4. **验证**（[问题排查2、3](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#2)）

   通过`lsmod | grep dcu`命令验证驱动是否安装成功，如图所示：

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/qudongjianceunbuntu.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/qudongjianceunbuntu.png)

## 第三步：环境部署

*注：非root用户请务必**加入39组**，才能正确调用DCU加速卡，通过命令`usermod -a -G 39 $USER`完成设置。*

### **容器化部署方式（推荐）**

##### ➡rpm系列系统（CentOS，NFS，Anolis，Kylin，UOS，openEuler）

1. **安装docker**

   *注：[Docker](https://docs.docker.com/) 要求内核版本不低于 3.10，建议安装docker-19.03以上版本；若安装失败，建议使用docker国内源。*

   ```bash
   yum install -y docker-ce docker-ce-cli containerd.io \
   && systemctl daemon-reload \
   && systemctl restart docker
   ```

2. **获取镜像**

   *注：镜像获取可以前往镜像仓库—[光源](https://sourcefind.cn/)，挑选所需**DCU**镜像，复制相应带有**latest**标签的命令，并在命令行执行，例如拉取pytorch镜像。*

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/guangquan.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/guangquan.png)

   ```bash
   docker pull image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.10.0-centos7.6-dtk-22.10-py38-latest
   ```

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/jinxianglaqu.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/jinxianglaqu.png)

3. **启动容器环境**

   *注：该启动参数可根据实际情况进行删减，参数如下：*

   *-i         打开容器标准输入*

   *-t        分配一个伪终端*

   *-v       挂载数据卷*

   *--network  连接网络（none|host|自定义网络...）*

   *--name   为容器添加名字*

   *--ipc     设置IPC模式（none|shareable|host...）*

   *--shm-size    设置/dev/shm大小*

   *--group-add    设置用户附加组（DCU需要添加39组）*

   *--device     指定访问设备（DCU需要添加/dev/kfd以及/dev/dri）*

   *--cap-add     添加权限（SYS_PTRACE|NET_ADMIN...）*

   *--security-opt 安全配置（seccomp=unconfined|label=disable...）*

   *--privileged 特权模式*

   *...*

   ```bash
   docker run \
   -it \
   --name=test \
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

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/jinrudockerpy83.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/jinrudockerpy83.png)

##### ➡deb系列系统（Ubuntu）

1. **安装docker**

   *注：推荐使用ubuntu的LTS版，建议安装docker-19.03以上版本；若安装失败，建议使用docker国内源。或者下载脚本`curl -fsSL get.docker.com -o get-docker.sh`，简化安装流程。*

   ```bash
   apt-get install -y docker-ce docker-ce-cli containerd.io \
   && systemctl daemon-reload \
   && systemctl restart docker
   ```

2. [获取镜像](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#pull)（同rpm系列系统，选择带有**ubuntu**名称的镜像即可）

3. [启动容器环境](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#run)（同rpm系列系统）

### **物理机部署方式**

##### ➡rpm系列系统（CentOS，NFS，Anolis，Kylin，UOS，openEuler）

1. **安装DTK**（DCU Toolkit，DCU软件平台）依赖包（[问题排查4](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#4)）

   ```bash
   yum install -y \
       epel-release \
       centos-release-scl \
       && yum clean all \
       && yum makecache \
   && yum groupinstall -y "Development tools" \
   && yum install -y \
       vim \
       curl \
       bzip2 \
       bzip2-devel \
       sudo \
       gcc \
       uuid-devel \
       gdbm-devel \
       readline-devel \
       tk-devel \
       openssl \
       openssl-devel \
       openssl-static \
       rpm-build \
       patch \
       ninja-build \
       glog-devel \
       lmdb-devel \
       opencv-devel \
       openblas-devel \
       libibverbs-devel \
       gflags-devel \
       gstreamer1 \
       gstreamer1-devel \
       gstreamer1-plugins-base \
       gstreamer1-plugins-base-devel \
       gstreamer1-plugins-bad-free \
       gstreamer1-plugins-bad-free-devel \
       gstreamer1-plugins-good \
       gstreamer1-plugins-ugly-free \
       gstreamer1-plugins-ugly-free-devel \
       gst123 \
       libibverbs-devel \
       libibverbs-utils \
       libffi-devel \
       zlib-devel \
       openssl-devel \
       ncurses-devel \
       sqlite-devel \
       devtoolset-7-gcc* \
       numactl \
       numactl-devel \
       wget \
       openssh \
       openssh-server
   ```

2. **python安装**

   ```bash
   cd /tmp \
   && wget -O python.tgz https://registry.npmmirror.com/-/binary/python/3.8.12/Python-3.8.12.tgz \
   && mkdir python-tmp \
   && tar -xvf python.tgz -C ./python-tmp --strip-components 1 \
   && cd python-tmp \
   && ./configure \
       --enable-shared \
   && make -j$(nproc) \
   && make install \
   && rm -rf /tmp/python* \
   && ln -s /usr/local/bin/python3 /usr/local/bin/python \
   && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip
   ```

3. **pip更新**

   ```bash
   pip install --no-cache-dir --upgrade pip
   ```

4. **cmake安装**

   ```bash
   cd /tmp \
   && wget -O cmake.tar.gz https://cmake.org/files/v3.19/cmake-3.19.3-Linux-x86_64.tar.gz \
   && mkdir /opt/cmake \
   && tar -xvf cmake.tar.gz -C /opt/cmake --strip-components 1 \
   && rm -rf /tmp/cmake*
   ```

5. **hwloc安装**

   ```bash
   cd /tmp \
   && wget -O hwloc.tar.gz https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.1.tar.gz \
   && mkdir hwloc-tmp \
   && tar -xvf hwloc.tar.gz -C ./hwloc-tmp --strip-components 1 \
   && cd hwloc-tmp \
   && ./configure --prefix=/opt/hwloc \
   && make -j$(nproc) \
   && make install \
   && rm -rf /tmp/hwloc*
   ```

6. **mpi安装**

   ```bash
   cd /tmp \
   && wget -O openmpi.tar.gz https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz \
   && mkdir openmpi-tmp \
   && tar -xvf openmpi.tar.gz -C ./openmpi-tmp --strip-components 1 \
   && cd openmpi-tmp \
   && ./configure \
      --prefix=/opt/mpi/ \
      --with-hwloc=/opt/hwloc/ \
      --enable-orterun-prefix-by-default \
      --enable-mpi-thread-multiple \
      --enable-dlopen \
   && make -j$(nproc) \
   && make install \
   && rm -rf /tmp/openmpi*
   ```

7. **获取DTK**

   *注：可前往[开发者社区](https://developer.hpccube.com/)→**资源工具**→**DCU Toolkit**，获取**latest** DTK下载地址。*

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/dcutoolkit.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/dcutoolkit.png)

8. **安装DTK**

   ```bash
   tar -xvf DTK-*.tar.gz -C /opt/ \
   && ln -s /opt/dtk-* /opt/dtk
   ```

9. **设置环境变量**

   ```bash
   cat > /etc/profile.d/dtk.sh <<-"EOF"
   #!/bin/bash
   
   #gcc
   source /opt/rh/devtoolset-7/enable
   
   #python3
   export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/lib64/:$LD_LIBRARY_PATH
   export PATH=/usr/local/bin:$PATH
   export PYTHONPATH=/usr/local/:$PYTHONPATH
   
   #cmake
   export PATH=/opt/cmake/bin/:$PATH
   
   #hwloc
   export PATH=/opt/hwloc/bin/:${PATH} \
   export LD_LIBRARY_PATH=/opt/hwloc/lib:${LD_LIBRARY_PATH}
   
   #mpi
   export LD_LIBRARY_PATH=/opt/mpi/lib:$LD_LIBRARY_PATH
   export PATH=/opt/mpi/bin:$PATH
   
   #DTK
   source /opt/dtk/env.sh
   EOF
   source /etc/profile.d/dtk.sh
   ```

10. **验证DCU环境**（[问题排查5、6](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#5)）

    通过`rocm-smi`以及`rocminfo | grep gfx`命令验证DCU环境安装完毕，如图所示：

    [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocm-smi.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocm-smi.png)

    [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocminfogrepgfx.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocminfogrepgfx.png)

##### ➡deb系列系统（Ubuntu）

1. **安装DTK**（DCU Toolkit，DCU软件平台）依赖包（[问题排查4](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#4)）

   ```bash
   apt-get update -y \
       && apt-get install --no-install-recommends -y \
          build-essential \
              git \
              wget \
              gfortran \
              elfutils \
              libelf-dev \
              libdrm-dev \
              kmod \
              libtinfo5 \
              sqlite3 \
              libsqlite3-dev \
              libnuma-dev \
              libgl1-mesa-dev \
              alien \
              rsync \
              libpci-dev \
              pciutils \
              libpciaccess-dev \
              libbabeltrace-dev \
              pkg-config \
              libfile-which-perl \
              libfile-basedir-perl \
              libfile-copy-recursive-perl \
              libfile-listing-perl \
              libprotobuf-dev \
              libio-digest-perl \
              libdigest-md5-file-perl \
              libdata-dumper-simple-perl \
              vim \
              curl \
              libcurlpp-dev \
              openssh-server \
              sudo \
              locales \
              openssl \
              libssl-dev \
              patch \
              ninja-build \
              libgoogle-glog-dev \
              liblmdb-dev \
              libopenblas-dev \
              libgflags-dev \
              libibverbs-dev \
              ibverbs-utils \
              libffi-dev \
              zlib1g \
              zlib1g-dev \
              libbz2-dev \
              libncurses-dev \
              libsqlite3-dev \
              read-edid \
              numactl \
              libjpeg62 \
              liblzma-dev \
              libgdbm-dev \
              libgdbm-compat-dev \
              libnss3-dev \
              libreadline-dev \
              libncurses5-dev \
              libncursesw5-dev \
              xz-utils \
              tk-dev \
          && apt-get clean \
          && rm -rf /var/lib/apt/lists/*
   ```

2. **python安装**

   ```bash
   cd /tmp \
   && wget -O python.tgz https://registry.npmmirror.com/-/binary/python/3.8.12/Python-3.8.12.tgz \
   && mkdir python-tmp \
   && tar -xvf python.tgz -C ./python-tmp --strip-components 1 \
   && cd python-tmp \
   && ./configure \
       --enable-shared \
   && make -j$(nproc) \
   && make install \
   && rm -rf /tmp/python* \
   && ln -s /usr/local/bin/python3 /usr/local/bin/python \
   && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip \
   && apt-get update -y \
       && apt-get install --no-install-recommends -y \
          libgstreamer1.0-dev \
          libgstreamer-plugins-base1.0-dev \
          libgstreamer-plugins-bad1.0-dev \
          gstreamer1.0-plugins-base \
          gstreamer1.0-plugins-good \
          gstreamer1.0-plugins-bad \
          gstreamer1.0-plugins-ugly \
          gstreamer1.0-libav \
          gstreamer1.0-tools \
          gstreamer1.0-x \
          gstreamer1.0-alsa \
          gstreamer1.0-gl \
          gstreamer1.0-gtk3 \
          gstreamer1.0-qt5 \
          gstreamer1.0-pulseaudio \
          gst123 \
          libopencv-dev \
          python3-opencv \
       && apt-get clean \
       && rm -rf /var/lib/apt/lists/*
   ```

3. **pip更新**

   ```bash
   pip install --no-cache-dir --upgrade pip
   ```

4. **cmake安装**

   ```bash
   cd /tmp \
   && wget -O cmake.tar.gz https://cmake.org/files/v3.19/cmake-3.19.3-Linux-x86_64.tar.gz \
   && mkdir /opt/cmake \
   && tar -xvf cmake.tar.gz -C /opt/cmake --strip-components 1 \
   && rm -rf /tmp/cmake*
   ```

5. **hwloc安装**

   ```bash
   cd /tmp \
   && wget -O hwloc.tar.gz https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.1.tar.gz \
   && mkdir hwloc-tmp \
   && tar -xvf hwloc.tar.gz -C ./hwloc-tmp --strip-components 1 \
   && cd hwloc-tmp \
   && ./configure --prefix=/opt/hwloc \
   && make -j$(nproc) \
   && make install \
   && rm -rf /tmp/hwloc*
   ```

6. **mpi安装**

   ```bash
   cd /tmp \
   && wget -O openmpi.tar.gz https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz \
   && mkdir openmpi-tmp \
   && tar -xvf openmpi.tar.gz -C ./openmpi-tmp --strip-components 1 \
   && cd openmpi-tmp \
   && ./configure \
      --prefix=/opt/mpi/ \
      --with-hwloc=/opt/hwloc/ \
      --enable-orterun-prefix-by-default \
      --enable-mpi-thread-multiple \
      --enable-dlopen \
   && make -j$(nproc) \
   && make install \
   && rm -rf /tmp/openmpi*
   ```

7. **获取DTK**

   *注：可前往[开发者社区](https://developer.hpccube.com/)→**资源工具**→**DCU Toolkit**，获取**latest** DTK下载地址。*

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/dcutoolkit.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/dcutoolkit.png)

8. **安装DTK**

   ```bash
   tar -xvf DTK-*.tar.gz -C /opt/ \
   && ln -s /opt/dtk-* /opt/dtk
   ```

9. **设置环境变量**

   ```bash
   cat > /etc/profile.d/dtk.sh <<-"EOF"
   #!/bin/bash
   
   #python3
   export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/lib64/:$LD_LIBRARY_PATH
   export PATH=/usr/local/bin:$PATH
   export PYTHONPATH=/usr/local/:$PYTHONPATH
   
   #cmake
   export PATH=/opt/cmake/bin/:$PATH
   
   #hwloc
   export PATH=/opt/hwloc/bin/:${PATH} \
   export LD_LIBRARY_PATH=/opt/hwloc/lib:${LD_LIBRARY_PATH}
   
   #mpi
   export LD_LIBRARY_PATH=/opt/mpi/lib:$LD_LIBRARY_PATH
   export PATH=/opt/mpi/bin:$PATH
   
   #DTK
   source /opt/dtk/env.sh
   EOF
   source /etc/profile.d/dtk.sh
   ```

10. **验证DCU环境**（[问题排查5、6](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#5)）

    通过`rocm-smi`以及`rocminfo | grep gfx`命令验证DCU环境安装完毕，如图所示：

    [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocm-smi.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocm-smi.png)

    [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocminfogrepgfx.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/rocminfogrepgfx.png)

## 第四步：实例测试

*注：`HIP_VISIBLE_DEVICES`—设置DCU加速卡可见性，类似CUDA_VISIBLE_DEVICES。例如：`export HIP_VISIBLE_DEVICES=0`表示设置第一块DCU可见。*

### 训练测试

1. **训练环境搭建**

   *注：若在[光源](https://sourcefind.cn/)拉取相应框架镜像后则**无需搭建环境**，进入后即得到完整训练环境，可直接跳转至[环境验证](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#train)（**推荐**）。本次环境搭建以Pytorch框架为例，各类框架（pytorch，tensorflow，paddlepaddle，oneflow...）请前往[开发者社区](https://developer.hpccube.com/)→**资源工具**→**AI 生态包**下获取最新whl包并安装。*

   - 下载Pytorch以及TorchVision框架包

     *注：各种框架的whl包应与上述安装过程中**DTK版本**对应，例如已安装dtk-22.10，则需要到**AI 生态包**→**pytorch**→**dtk-22.10**以及**AI 生态包**→**vision**→**dtk-22.10**文件夹中下载whl包并安装。*

     [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kafazheshequaibao.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kafazheshequaibao.png)

   - 安装Pytorch以及TorchVision（[问题排查7](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#7)）

     ```bash
     pip3 install torch-*.whl \
     && pip3 install torchvision-*.whl \
     && pip3 install numpy
     ```

2. **环境验证**（[问题排查8](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#8)）

   ```bash
   python3 -c "import torch;print('pytorch version:',torch.__version__);print('DCU is',torch.cuda.is_available())"
   ```

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/keyongjiance.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/keyongjiance.png)

3. **训练代码获取**

   *注：可前往[ModelZoo](https://sourcefind.cn/#/model-zoo/list)获取**实例代码**，本次以flavr_pytorch为例；同时，**DCU也兼容GPU开源深度学习代码**。*

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/flavr_pytorch_.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/flavr_pytorch_.png)

4. **克隆训练代码**

   ```bash
   git clone http://developer.hpccube.com/codes/modelzoo/flavr_pytorch.git
   ```

5. **代码依赖安装**

   ```bash
   cd flavr_pytorch \
   && pip install -r requirements.txt
   ```

6. **启动测试**

   ```powershell
   python main.py --batch_size 32 --test_batch_size 32 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root datasets --n_outputs 1 --num_gpu 1
   ```

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/flavr_pytorch.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/flavr_pytorch.png)

### 推理测试

1. **推理环境搭建**

   *注：若在[光源](https://sourcefind.cn/)拉取相应框架镜像后则**无需搭建环境**，进入后即得到完整推理环境，可直接跳转至[环境验证](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#infer)（**推荐**）。本次环境搭建以MIGraphX框架为例。*

   - 安装half

     ```bash
     wget https://github.com/ROCmSoftwarePlatform/half/archive/1.12.0.tar.gz \
     && tar -xvf 1.12.0.tar.gz \
     && cp half-1.12.0/include/half.hpp /opt/dtk/include/
     ```

   - 安装sqlite

     ```bash
     cd /tmp \
     && wget --no-cookie --no-check-certificate -O sqlite.tar.gz https://www.sqlite.org/2023/sqlite-autoconf-3410000.tar.gz \
     && mkdir sqlite-tmp \
     && tar -xvf sqlite.tar.gz -C ./sqlite-tmp --strip-components 1 \
     && cd sqlite-tmp \
     && ./configure \
     && make -j$(nproc) \
     && make install \
     && cd \
     && rm -rf /tmp/sqlite*
     ```

   - 下载MIGraphX

     *注：MIGraphX框架安装包应与安装过程中**系统版本**以及**DTK版本**对应，例如centos7.6系统中安装dtk-23.10.1，则需要到**AI 生态包**→**migraphx**→**dtk-23.10.1**→**CentOS7.6**文件夹下载rpm安装包并安装。*

     [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kafazheshequaibao.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kafazheshequaibao.png)

   - 安装MIGraphX

     ```bash
     rpm -ivh migraphx-*.rpm --nodeps --force \
     && rpm -ivh migraphx-devel-*.rpm --nodeps --force
     ```

   - 设置环境变量

     ```bash
     cat > /etc/profile.d/migraphx.sh <<-"EOF"
     #!/bin/bash
     
     export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
     export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
     export PYTHONPATH=/opt/dtk/lib:$PYTHONPATH
     EOF
     source /etc/profile.d/migraphx.sh
     ```

2. **环境验证**

   ```bash
   /opt/dtk/bin/migraphx-driver onnx -l
   ```

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/onnx_test.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/onnx_test.png)

3. **推理代码获取**

   *注：可前往[ModelZoo](https://sourcefind.cn/#/model-zoo/list)获取**实例代码**，本次以yolov7_migraphx为例。*

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/yolov7_migraphx_.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/yolov7_migraphx_.png)

4. **克隆推理代码**

   ```bash
   git clone http://developer.hpccube.com/codes/modelzoo/yolov7_migraphx.git
   ```

5. **代码依赖安装**

   ```bash
   cd yolov7_migraphx/Python/ \
   && pip install -r requirements.txt
   ```

6. **启动测试**

   ```bash
   python YoloV7_infer_migraphx.py
   ```

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/yolov7_test.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/yolov7_test.png)

   [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/Result.jpg)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/Result.jpg)

# 常用资源

[1] [开发者社区](https://developer.hpccube.com/)—驱动、DTK、框架、代码等资源下载中心以及学习中心

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kaifazequanmao.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/kaifazequanmao.png)

[2] [光源](https://sourcefind.cn/)—docker镜像获取中心

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/guangyuan.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/guangyuan.png)

[3] [开发者论坛](https://forum.hpccube.com/)—问题讨论中心

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/forume2.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/forume2.png)

[4] [DCU FAQ](https://developer.hpccube.com/gitbook//dcu_faq/index.html)—DCU常见问题解答

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/faq1.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/faq1.png)

# 问题排查

**1.问：**`lspci | grep -i Display`无显示

> **答**：清理DCU加速卡金手指，确保各插槽插紧无松动。

**2.问：**`lsmod | grep hydcu`无显示

> **答**：请先执行驱动安装步骤；若仍无显示，通过命令`modprobe hydcu`手动加载驱动并重启机器；若驱动仍未加载，请查看/etc/modprobe.d/hydcu.conf是否存在，不存在可通过命令`echo “options hydcu hygon_vbios=0” > /etc/modprobe.d/hydcu.conf`手动创建。

**3.问：** 驱动加载失败，如下图：

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/shibeicuowu.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/shibeicuowu.png)

> **答**：请检查系统启动项中是否包含**nomodeset**选项，若存在，请删除。通常在系统启动时，按 **e**进入内核启动修改页面，找到以 **linux16 /vmlinuz** 开始的行，删除 **nomodeset** 字段，然后按**Ctrl+x**启动，如图所示：
>
> [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/qidongx.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/qidongx.png)
>
> 在系统启动之后，根据不同版本的系统要求修改 grub 文件，确保该启动项永久生效。

**4.问：** 某些依赖包无法安装

> **答**：建议换国内源。其中**lmdb-devel、glog-devel、opencv-devel、openblas-devel、gflags-devel、gstreamer**等，或**liblmdb-dev、libopenblas-dev、 libgflags-dev、libopencv-dev、gstreamer**等无法安装，无需担心，该依赖包已导入框架内；其中若**devtoolset-7**安装失败，可以进入DCU Toolkit下载后离线安装：
>
> [![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/dcutoolkit.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/dcutoolkit.png)

```bash
mkdir /opt/rh \
&& tar xvf devtoolset-7.3.1.tar.gz -C /opt/rh
```

**5.问：** `rocm-smi`显示正常，`rocminfo`出现如下图报错：

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/cuowurocminfo.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/cuowurocminfo.png)

> **答**：请通过命令`usermod -a -G 39 $USER`将用户加入**39**组。

**6.问：**`rocm-smi`、`rocminfo`命令都查找不到

> **答**：请**根据系统**执行DTK安装中的**[设置环境变量](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html#env)**步骤。

**7.问：**pip install 安装出现如图不支持提示

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/pipnotsup.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/pipnotsup.png)

> **答**：请通过命令`pip install --upgrade pip`将pip升级，若未解决，请查看下载的whl包py版本是否与环境中的python版本一致，以及检查环境中是否已正确安装DTK软件包。

**8.问：**使用期间出现如下导入hsa相关库报错

[![img](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/importtorcherror.png)](https://developer.sourcefind.cn/gitbook//dcu_tutorial/images/importtorcherror.png)

> **答**：dtk23.10系列设计如此，请在启动容器时挂载hyhal：-v /opt/hyhal:/opt/hyhal，或在开发者社区下载[hyhal](https://cancon.hpccube.com:65024/directlink/1/latest/hyhal.tar.gz)，放入容器/opt/下，并解压。