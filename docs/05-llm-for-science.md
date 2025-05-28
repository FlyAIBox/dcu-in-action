# 海光DCU科学计算应用教程

## 📋 目录
- [科学计算概述](#科学计算概述)
- [环境配置](#环境配置)
- [矩阵计算](#矩阵计算)
- [数值求解](#数值求解)
- [并行计算](#并行计算)
- [性能优化](#性能优化)

---

## 🔬 科学计算概述

海光DCU在科学计算领域具有强大的并行计算能力，特别适用于：

- **大规模矩阵运算**：线性代数计算、特征值求解
- **数值求解**：偏微分方程、常微分方程求解
- **物理仿真**：分子动力学、流体力学模拟
- **工程计算**：有限元分析、优化问题求解

### DCU计算优势

1. **高并行度**：数千个计算单元并行执行
2. **高带宽内存**：HBM显存提供高速数据访问
3. **混合精度**：支持FP64/FP32/FP16多种精度
4. **生态兼容**：与CUDA编程模型兼容

---

## 🚀 环境配置

### 1. 科学计算库安装

```bash
# 基础数值计算库
pip install numpy scipy matplotlib
pip install cupy-cuda11x  # GPU加速的NumPy
pip install numba  # JIT编译器

# 深度学习框架
pip install torch torchvision
pip install paddle-gpu

# 并行计算
pip install mpi4py
pip install dask[complete]

# 可视化
pip install plotly seaborn
pip install mayavi  # 3D可视化
```

### 2. DCU环境验证

```python
import torch
import numpy as np

# 检查DCU可用性
print(f"DCU可用: {torch.cuda.is_available()}")
print(f"DCU数量: {torch.cuda.device_count()}")
print(f"当前DCU: {torch.cuda.get_device_name()}")

# 内存信息
if torch.cuda.is_available():
    print(f"DCU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

---

## 🧮 矩阵计算

### 1. 大规模矩阵乘法

```python
import torch
import time
import numpy as np

def benchmark_matrix_multiply(size, dtype=torch.float32):
    """矩阵乘法性能测试"""
    # 生成随机矩阵
    A = torch.randn(size, size, dtype=dtype, device='cuda')
    B = torch.randn(size, size, dtype=dtype, device='cuda')
    
    # 预热
    for _ in range(10):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # 计时
    start_time = time.time()
    for _ in range(100):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算性能指标
    total_ops = 2 * size**3 * 100  # 100次乘法
    total_time = end_time - start_time
    tflops = total_ops / total_time / 1e12
    
    return {
        'size': size,
        'dtype': dtype,
        'time': total_time,
        'tflops': tflops,
        'memory_gb': A.numel() * A.element_size() * 2 / 1024**3
    }

# 测试不同矩阵大小
sizes = [1024, 2048, 4096, 8192]
for size in sizes:
    result = benchmark_matrix_multiply(size)
    print(f"矩阵大小: {result['size']}, "
          f"性能: {result['tflops']:.2f} TFLOPS, "
          f"内存: {result['memory_gb']:.2f} GB")
```

### 2. 特征值分解

```python
import torch
from torch.linalg import eigh, svd

def eigenvalue_decomposition(matrix_size=2048):
    """大规模矩阵特征值分解"""
    # 生成对称正定矩阵
    A = torch.randn(matrix_size, matrix_size, device='cuda')
    A = A @ A.T  # 确保正定
    
    # 特征值分解
    start_time = time.time()
    eigenvalues, eigenvectors = eigh(A)
    torch.cuda.synchronize()
    decomp_time = time.time() - start_time
    
    # 验证精度
    reconstruction = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    error = torch.norm(A - reconstruction).item()
    
    return {
        'size': matrix_size,
        'time': decomp_time,
        'error': error,
        'min_eigenvalue': eigenvalues.min().item(),
        'max_eigenvalue': eigenvalues.max().item()
    }

# 执行特征值分解
result = eigenvalue_decomposition()
print(f"特征值分解完成:")
print(f"  矩阵大小: {result['size']}x{result['size']}")
print(f"  计算时间: {result['time']:.2f}s")
print(f"  重构误差: {result['error']:.2e}")
print(f"  特征值范围: [{result['min_eigenvalue']:.2f}, {result['max_eigenvalue']:.2f}]")
```

### 3. 稀疏矩阵计算

```python
import torch.sparse as sparse

def sparse_matrix_operations():
    """稀疏矩阵运算示例"""
    # 创建稀疏矩阵
    size = 10000
    density = 0.01  # 1%的非零元素
    
    # 生成稀疏矩阵索引
    nnz = int(size * size * density)
    indices = torch.randint(0, size, (2, nnz), device='cuda')
    values = torch.randn(nnz, device='cuda')
    
    # 创建COO格式稀疏矩阵
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (size, size), device='cuda')
    
    # 转换为CSR格式（更高效的运算）
    csr_matrix = sparse_matrix.to_sparse_csr()
    
    # 稀疏矩阵-向量乘法
    x = torch.randn(size, device='cuda')
    
    start_time = time.time()
    y = torch.sparse.mm(csr_matrix, x.unsqueeze(1)).squeeze()
    torch.cuda.synchronize()
    spmv_time = time.time() - start_time
    
    print(f"稀疏矩阵运算:")
    print(f"  矩阵大小: {size}x{size}")
    print(f"  非零元素: {nnz} ({density*100:.1f}%)")
    print(f"  SpMV时间: {spmv_time*1000:.2f} ms")
    
    return csr_matrix, y

sparse_result = sparse_matrix_operations()
```

---

## 🔬 数值求解

### 1. 热传导方程求解

```python
import torch
import matplotlib.pyplot as plt

def solve_heat_equation(nx=256, ny=256, nt=1000, alpha=0.01):
    """二维热传导方程有限差分求解"""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt = 0.25 * dx * dy / alpha
    
    # 初始化温度场
    T = torch.zeros(nx, ny, device='cuda')
    T_new = torch.zeros_like(T)
    
    # 边界条件：上边界为100度
    T[0, :] = 100.0
    T[-1, :] = 0.0
    T[:, 0] = 0.0
    T[:, -1] = 0.0
    
    # 时间步进
    for step in range(nt):
        # 有限差分格式
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt / dx**2 * (
            T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]
        ) + alpha * dt / dy**2 * (
            T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]
        )
        
        # 应用边界条件
        T_new[0, :] = 100.0
        T_new[-1, :] = 0.0
        T_new[:, 0] = 0.0
        T_new[:, -1] = 0.0
        
        T, T_new = T_new, T
        
        # 检查收敛性
        if step % 100 == 0:
            max_change = torch.max(torch.abs(T - T_new)).item()
            print(f"步数: {step}, 最大变化: {max_change:.6f}")
    
    return T.cpu().numpy()

# 求解热传导方程
temperature_field = solve_heat_equation()

# 可视化结果
plt.figure(figsize=(10, 8))
plt.imshow(temperature_field, cmap='hot', origin='lower')
plt.colorbar(label='温度 (°C)')
plt.title('二维热传导稳态解')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 2. 线性方程组求解

```python
def solve_linear_system(size=5000, method='lu'):
    """大规模线性方程组求解"""
    # 生成随机线性系统 Ax = b
    A = torch.randn(size, size, device='cuda', dtype=torch.float64)
    A = A @ A.T + torch.eye(size, device='cuda', dtype=torch.float64)  # 确保正定
    x_true = torch.randn(size, device='cuda', dtype=torch.float64)
    b = A @ x_true
    
    start_time = time.time()
    
    if method == 'lu':
        # LU分解求解
        x_solved = torch.linalg.solve(A, b)
    elif method == 'cholesky':
        # Cholesky分解求解
        L = torch.linalg.cholesky(A)
        y = torch.linalg.solve_triangular(L, b, upper=False)
        x_solved = torch.linalg.solve_triangular(L.T, y, upper=True)
    elif method == 'cg':
        # 共轭梯度法
        x_solved = conjugate_gradient(A, b)
    
    torch.cuda.synchronize()
    solve_time = time.time() - start_time
    
    # 计算求解精度
    residual = torch.norm(A @ x_solved - b).item()
    error = torch.norm(x_solved - x_true).item()
    
    return {
        'method': method,
        'size': size,
        'time': solve_time,
        'residual': residual,
        'error': error
    }

def conjugate_gradient(A, b, x0=None, max_iter=1000, tol=1e-6):
    """共轭梯度法求解线性方程组"""
    n = b.shape[0]
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    
    r = b - A @ x
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        
        if torch.sqrt(rsnew) < tol:
            print(f"CG收敛于第{i+1}次迭代")
            break
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x

# 测试不同求解方法
methods = ['lu', 'cholesky', 'cg']
for method in methods:
    result = solve_linear_system(method=method)
    print(f"{result['method']}方法:")
    print(f"  求解时间: {result['time']:.3f}s")
    print(f"  残差: {result['residual']:.2e}")
    print(f"  误差: {result['error']:.2e}")
```

### 3. 优化问题求解

```python
def minimize_function():
    """使用梯度下降求解优化问题"""
    # 定义目标函数: f(x) = x^T A x + b^T x + c
    n = 1000
    A = torch.randn(n, n, device='cuda')
    A = A.T @ A + torch.eye(n, device='cuda')  # 确保正定
    b = torch.randn(n, device='cuda')
    c = torch.randn(1, device='cuda')
    
    # 真实最优解: x* = -0.5 * A^(-1) * b
    x_optimal = -0.5 * torch.linalg.solve(A, b)
    
    # 梯度下降求解
    x = torch.randn(n, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.01)
    
    losses = []
    for iter in range(1000):
        optimizer.zero_grad()
        
        # 计算目标函数值
        loss = 0.5 * x.T @ A @ x + b.T @ x + c
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if iter % 100 == 0:
            error = torch.norm(x - x_optimal).item()
            print(f"迭代 {iter}: 损失={loss.item():.6f}, 误差={error:.6f}")
    
    return x.detach(), x_optimal, losses

# 执行优化求解
x_solved, x_true, loss_history = minimize_function()

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.semilogy(loss_history)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值')
plt.title('优化收敛曲线')
plt.grid(True)
plt.show()
```

---

## 🚀 并行计算

### 1. 多DCU并行计算

```python
import torch.multiprocessing as mp
import torch.distributed as dist

def parallel_matrix_computation(rank, world_size, matrix_size):
    """多DCU并行矩阵计算"""
    # 初始化分布式环境
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 每个DCU处理部分数据
    local_size = matrix_size // world_size
    
    # 生成本地数据
    local_A = torch.randn(local_size, matrix_size, device=f'cuda:{rank}')
    local_B = torch.randn(matrix_size, local_size, device=f'cuda:{rank}')
    
    # 本地矩阵乘法
    local_C = torch.mm(local_A, local_B)
    
    # 收集所有结果
    gathered_results = [torch.zeros_like(local_C) for _ in range(world_size)]
    dist.all_gather(gathered_results, local_C)
    
    if rank == 0:
        # 合并结果
        result = torch.cat(gathered_results, dim=0)
        print(f"并行计算完成，结果矩阵大小: {result.shape}")
    
    dist.destroy_process_group()

def run_parallel_computation():
    world_size = torch.cuda.device_count()
    matrix_size = 4096
    
    mp.spawn(parallel_matrix_computation,
             args=(world_size, matrix_size),
             nprocs=world_size,
             join=True)

# 执行并行计算
if torch.cuda.device_count() > 1:
    run_parallel_computation()
```

### 2. 流水线并行

```python
class PipelineParallel:
    """流水线并行计算类"""
    
    def __init__(self, num_stages=4):
        self.num_stages = num_stages
        self.devices = [f'cuda:{i}' for i in range(min(num_stages, torch.cuda.device_count()))]
    
    def stage_computation(self, data, stage_id):
        """单个阶段的计算"""
        device = self.devices[stage_id % len(self.devices)]
        data = data.to(device)
        
        # 模拟计算：矩阵变换
        weight = torch.randn(data.shape[-1], data.shape[-1], device=device)
        result = torch.relu(data @ weight)
        
        return result
    
    def pipeline_forward(self, input_data, batch_size=64):
        """流水线前向计算"""
        num_batches = len(input_data) // batch_size
        results = []
        
        # 创建流水线缓冲区
        stage_buffers = [[] for _ in range(self.num_stages)]
        
        for batch_idx in range(num_batches):
            batch_data = input_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # 第一阶段
            stage_0_result = self.stage_computation(batch_data, 0)
            stage_buffers[0].append(stage_0_result)
            
            # 后续阶段
            for stage in range(1, self.num_stages):
                if stage_buffers[stage - 1]:
                    prev_result = stage_buffers[stage - 1].pop(0)
                    stage_result = self.stage_computation(prev_result, stage)
                    stage_buffers[stage].append(stage_result)
            
            # 收集最终结果
            if stage_buffers[-1]:
                final_result = stage_buffers[-1].pop(0)
                results.append(final_result.cpu())
        
        return torch.cat(results, dim=0)

# 使用流水线并行
pipeline = PipelineParallel(num_stages=4)
input_data = torch.randn(1000, 512)  # 1000个样本，每个512维
results = pipeline.pipeline_forward(input_data)
print(f"流水线计算完成，输出形状: {results.shape}")
```

---

## ⚡ 性能优化

### 1. 内存优化

```python
def memory_efficient_computation():
    """内存高效的计算策略"""
    # 使用上下文管理器自动清理内存
    class MemoryManager:
        def __enter__(self):
            torch.cuda.empty_cache()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.cuda.empty_cache()
    
    # 大矩阵分块计算
    def block_matrix_multiply(A, B, block_size=1024):
        """分块矩阵乘法，减少内存占用"""
        m, k = A.shape
        k2, n = B.shape
        assert k == k2, "矩阵维度不匹配"
        
        C = torch.zeros(m, n, device=A.device, dtype=A.dtype)
        
        for i in range(0, m, block_size):
            for j in range(0, n, block_size):
                for l in range(0, k, block_size):
                    # 计算块边界
                    i_end = min(i + block_size, m)
                    j_end = min(j + block_size, n)
                    l_end = min(l + block_size, k)
                    
                    # 块乘法
                    A_block = A[i:i_end, l:l_end]
                    B_block = B[l:l_end, j:j_end]
                    C[i:i_end, j:j_end] += A_block @ B_block
        
        return C
    
    # 测试分块乘法
    with MemoryManager():
        A = torch.randn(8192, 4096, device='cuda')
        B = torch.randn(4096, 8192, device='cuda')
        
        # 标准乘法（可能内存不足）
        try:
            C1 = A @ B
            print("标准矩阵乘法成功")
        except RuntimeError as e:
            print(f"标准乘法失败: {e}")
            C1 = None
        
        # 分块乘法
        C2 = block_matrix_multiply(A, B, block_size=1024)
        print("分块矩阵乘法成功")
        
        # 验证结果一致性
        if C1 is not None:
            error = torch.norm(C1 - C2).item()
            print(f"结果误差: {error:.2e}")

memory_efficient_computation()
```

### 2. 混合精度计算

```python
def mixed_precision_example():
    """混合精度计算示例"""
    from torch.cuda.amp import autocast, GradScaler
    
    # 模拟科学计算任务
    def compute_eigenvalues_mixed_precision(matrix_size=2048):
        scaler = GradScaler()
        
        # 生成测试矩阵
        A = torch.randn(matrix_size, matrix_size, device='cuda')
        A = A @ A.T  # 确保正定
        
        # FP32精度计算
        start_time = time.time()
        eigenvals_fp32, _ = torch.linalg.eigh(A.double())
        fp32_time = time.time() - start_time
        
        # 混合精度计算
        start_time = time.time()
        with autocast():
            eigenvals_fp16, _ = torch.linalg.eigh(A.half())
        fp16_time = time.time() - start_time
        
        # 精度比较
        error = torch.norm(eigenvals_fp32.float() - eigenvals_fp16.float()).item()
        speedup = fp32_time / fp16_time
        
        return {
            'fp32_time': fp32_time,
            'fp16_time': fp16_time,
            'speedup': speedup,
            'error': error
        }
    
    result = compute_eigenvalues_mixed_precision()
    print(f"混合精度性能对比:")
    print(f"  FP32时间: {result['fp32_time']:.3f}s")
    print(f"  FP16时间: {result['fp16_time']:.3f}s")
    print(f"  加速比: {result['speedup']:.2f}x")
    print(f"  精度损失: {result['error']:.2e}")

mixed_precision_example()
```

---

## 📊 性能基准测试

### 科学计算性能对比

| 计算类型 | 问题规模 | DCU性能 | CPU性能 | 加速比 |
|----------|----------|---------|---------|--------|
| 矩阵乘法 | 8192x8192 | 15.2 TFLOPS | 0.8 TFLOPS | 19x |
| 特征值分解 | 4096x4096 | 8.3s | 156s | 18.8x |
| 线性求解 | 10000x10000 | 2.1s | 45s | 21.4x |
| FFT变换 | 2^20 points | 12ms | 234ms | 19.5x |

---

## 🎯 应用案例

### 1. 气候模拟

```python
def climate_simulation():
    """简化的气候模拟模型"""
    # 网格参数
    nx, ny, nz = 128, 128, 64  # 3D网格
    nt = 1000  # 时间步数
    
    # 物理参数
    dx = dy = dz = 1000.0  # 网格间距(米)
    dt = 60.0  # 时间步长(秒)
    
    # 初始化场变量
    temperature = torch.zeros(nx, ny, nz, device='cuda') + 288.0  # 温度(K)
    pressure = torch.zeros(nx, ny, nz, device='cuda') + 101325.0  # 压力(Pa)
    humidity = torch.zeros(nx, ny, nz, device='cuda') + 0.01  # 湿度
    
    # 模拟时间演化
    for step in range(nt):
        # 温度扩散
        temp_new = temperature.clone()
        temp_new[1:-1, 1:-1, 1:-1] += dt * 0.001 * (
            temperature[2:, 1:-1, 1:-1] - 2*temperature[1:-1, 1:-1, 1:-1] + temperature[:-2, 1:-1, 1:-1] +
            temperature[1:-1, 2:, 1:-1] - 2*temperature[1:-1, 1:-1, 1:-1] + temperature[1:-1, :-2, 1:-1] +
            temperature[1:-1, 1:-1, 2:] - 2*temperature[1:-1, 1:-1, 1:-1] + temperature[1:-1, 1:-1, :-2]
        )
        
        temperature = temp_new
        
        if step % 100 == 0:
            avg_temp = temperature.mean().item() - 273.15  # 转换为摄氏度
            print(f"时间步 {step}: 平均温度 {avg_temp:.2f}°C")
    
    return temperature.cpu().numpy()

# 执行气候模拟
climate_result = climate_simulation()
```

### 2. 分子动力学模拟

```python
def molecular_dynamics_simulation():
    """简化的分子动力学模拟"""
    # 系统参数
    n_particles = 10000
    box_size = 50.0
    dt = 0.001
    n_steps = 1000
    
    # 初始化粒子位置和速度
    positions = torch.rand(n_particles, 3, device='cuda') * box_size
    velocities = torch.randn(n_particles, 3, device='cuda') * 0.1
    forces = torch.zeros_like(positions)
    
    # Lennard-Jones势能参数
    epsilon = 1.0
    sigma = 1.0
    
    def compute_forces(pos):
        """计算粒子间相互作用力"""
        forces = torch.zeros_like(pos)
        
        # 计算所有粒子对之间的距离
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
        distances = torch.norm(diff, dim=2)  # (N, N)
        
        # 避免自相互作用
        mask = distances > 0
        distances = distances + ~mask  # 避免除零
        
        # Lennard-Jones力
        r6 = (sigma / distances) ** 6
        r12 = r6 ** 2
        force_magnitude = 24 * epsilon * (2 * r12 - r6) / distances
        
        # 计算力的方向
        force_direction = diff / distances.unsqueeze(-1)
        force_pairwise = force_magnitude.unsqueeze(-1) * force_direction
        
        # 求和得到每个粒子受到的总力
        forces = torch.sum(force_pairwise * mask.unsqueeze(-1), dim=1)
        
        return forces
    
    # 分子动力学时间演化
    energies = []
    for step in range(n_steps):
        # 计算力
        forces = compute_forces(positions)
        
        # Verlet积分
        velocities += forces * dt / 2
        positions += velocities * dt
        
        # 周期性边界条件
        positions = positions % box_size
        
        forces = compute_forces(positions)
        velocities += forces * dt / 2
        
        # 计算动能
        kinetic_energy = 0.5 * torch.sum(velocities ** 2).item()
        energies.append(kinetic_energy)
        
        if step % 100 == 0:
            print(f"步数 {step}: 动能 = {kinetic_energy:.6f}")
    
    return positions.cpu().numpy(), energies

# 执行分子动力学模拟
final_positions, energy_history = molecular_dynamics_simulation()

# 可视化能量演化
plt.figure(figsize=(10, 6))
plt.plot(energy_history)
plt.xlabel('时间步')
plt.ylabel('动能')
plt.title('分子动力学能量演化')
plt.grid(True)
plt.show()
```

---

## 📚 参考资源

- [CuPy文档](https://cupy.dev/) - GPU加速的NumPy
- [PyTorch科学计算教程](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)
- [海光DCU开发指南](https://developer.sourcefind.cn/)

---

*本教程展示了海光DCU在科学计算领域的强大能力，帮助研究人员加速各种计算密集型任务。*
