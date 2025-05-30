"""
HPC数值求解器
提供高性能科学计算、线性代数、微分方程求解、优化算法等功能
专门针对DCU硬件优化
"""

import time
import numpy as np
from typing import Callable, Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import scipy
    from scipy import sparse, optimize, integrate, linalg
    from scipy.sparse.linalg import spsolve, cg, gmres
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from ..utils.logger import get_logger, performance_monitor
from ..dcu.device_manager import DCUDeviceManager

logger = get_logger(__name__)


@dataclass
class SolverConfig:
    """求解器配置"""
    # 基础配置
    precision: str = "float64"  # float32, float64
    device: str = "auto"  # cpu, cuda, auto
    max_iterations: int = 1000
    tolerance: float = 1e-8
    
    # 并行配置
    use_parallel: bool = True
    num_workers: int = -1  # -1表示自动检测
    
    # 内存配置
    memory_efficient: bool = True
    chunk_size: int = 1024
    
    # 算法配置
    algorithm: str = "auto"
    preconditioner: str = "none"  # none, jacobi, ilu, amg
    
    # 调试配置
    verbose: bool = False
    save_history: bool = False


class BaseSolver(ABC):
    """求解器基类"""
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.device_manager = DCUDeviceManager()
        self.device = self._setup_device()
        self.history = []
        
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == "auto":
            if HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    
    def _to_device(self, array: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """将数组移动到指定设备"""
        if self.device == "cuda" and HAS_TORCH:
            return torch.from_numpy(array).cuda()
        return array
    
    def _to_numpy(self, array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """将数组转换为numpy格式"""
        if HAS_TORCH and torch.is_tensor(array):
            return array.cpu().numpy()
        return array
    
    @abstractmethod
    def solve(self, *args, **kwargs):
        """求解方法（子类需要实现）"""
        pass


class LinearSolver(BaseSolver):
    """线性方程组求解器"""
    
    @performance_monitor("线性方程组求解")
    def solve(self, A: np.ndarray, b: np.ndarray, method: str = "auto") -> Dict[str, Any]:
        """
        求解线性方程组 Ax = b
        
        Args:
            A: 系数矩阵
            b: 右端向量
            method: 求解方法 (direct, iterative, auto)
        
        Returns:
            包含解向量和求解信息的字典
        """
        start_time = time.time()
        
        # 检查输入
        if A.shape[0] != A.shape[1]:
            raise ValueError("系数矩阵必须是方阵")
        if A.shape[0] != b.shape[0]:
            raise ValueError("系数矩阵和右端向量维度不匹配")
        
        n = A.shape[0]
        logger.info(f"求解 {n}x{n} 线性方程组")
        
        # 自动选择方法
        if method == "auto":
            if n < 1000:
                method = "direct"
            else:
                method = "iterative"
        
        # 移动到计算设备
        if self.device == "cuda" and HAS_TORCH:
            A_device = self._to_device(A.astype(np.float32))
            b_device = self._to_device(b.astype(np.float32))
            
            if method == "direct":
                try:
                    x_device = torch.linalg.solve(A_device, b_device)
                    x = self._to_numpy(x_device)
                    residual_norm = np.linalg.norm(A @ x - b)
                    iterations = 1
                    converged = True
                except Exception as e:
                    logger.warning(f"GPU直接求解失败: {e}，回退到CPU")
                    method = "iterative"
            
            if method == "iterative":
                x, info = self._solve_iterative_gpu(A_device, b_device)
                residual_norm = info["residual_norm"]
                iterations = info["iterations"]
                converged = info["converged"]
                
        else:
            # CPU求解
            if method == "direct":
                if HAS_SCIPY:
                    try:
                        x = linalg.solve(A, b)
                        residual_norm = np.linalg.norm(A @ x - b)
                        iterations = 1
                        converged = True
                    except Exception as e:
                        logger.warning(f"直接求解失败: {e}，使用迭代方法")
                        method = "iterative"
                else:
                    x = np.linalg.solve(A, b)
                    residual_norm = np.linalg.norm(A @ x - b)
                    iterations = 1
                    converged = True
            
            if method == "iterative":
                x, info = self._solve_iterative_cpu(A, b)
                residual_norm = info["residual_norm"]
                iterations = info["iterations"]
                converged = info["converged"]
        
        solve_time = time.time() - start_time
        
        result = {
            "solution": x,
            "residual_norm": residual_norm,
            "iterations": iterations,
            "converged": converged,
            "solve_time": solve_time,
            "method": method,
            "matrix_size": n
        }
        
        if self.config.save_history:
            self.history.append(result)
        
        logger.info(f"求解完成: 方法={method}, 迭代次数={iterations}, "
                   f"残差={residual_norm:.2e}, 时间={solve_time:.3f}s")
        
        return result
    
    def _solve_iterative_gpu(self, A: torch.Tensor, b: torch.Tensor) -> Tuple[np.ndarray, Dict]:
        """GPU迭代求解"""
        # 使用共轭梯度法
        x = torch.zeros_like(b)
        r = b - torch.matmul(A, x)
        p = r.clone()
        rsold = torch.dot(r, r)
        
        iterations = 0
        converged = False
        
        for i in range(self.config.max_iterations):
            Ap = torch.matmul(A, p)
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            
            if torch.sqrt(rsnew) < self.config.tolerance:
                converged = True
                break
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
            iterations += 1
        
        x_numpy = self._to_numpy(x)
        residual_norm = float(torch.sqrt(rsnew).cpu())
        
        return x_numpy, {
            "residual_norm": residual_norm,
            "iterations": iterations,
            "converged": converged
        }
    
    def _solve_iterative_cpu(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """CPU迭代求解"""
        if HAS_SCIPY:
            # 使用SciPy的求解器
            x, info = cg(A, b, tol=self.config.tolerance, maxiter=self.config.max_iterations)
            converged = (info == 0)
            residual_norm = np.linalg.norm(A @ x - b)
            iterations = self.config.max_iterations if not converged else len(self.history) + 1
        else:
            # 简单的雅可比迭代
            x, info = self._jacobi_iteration(A, b)
            residual_norm = info["residual_norm"]
            iterations = info["iterations"]
            converged = info["converged"]
        
        return x, {
            "residual_norm": residual_norm,
            "iterations": iterations,
            "converged": converged
        }
    
    def _jacobi_iteration(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """雅可比迭代法"""
        n = A.shape[0]
        x = np.zeros(n)
        D = np.diag(A)
        R = A - np.diagflat(D)
        
        for i in range(self.config.max_iterations):
            x_new = (b - np.dot(R, x)) / D
            residual_norm = np.linalg.norm(A @ x_new - b)
            
            if residual_norm < self.config.tolerance:
                return x_new, {
                    "residual_norm": residual_norm,
                    "iterations": i + 1,
                    "converged": True
                }
            x = x_new
        
        return x, {
            "residual_norm": residual_norm,
            "iterations": self.config.max_iterations,
            "converged": False
        }


class ODESolver(BaseSolver):
    """常微分方程求解器"""
    
    @performance_monitor("ODE求解")
    def solve(self, func: Callable, y0: np.ndarray, t_span: Tuple[float, float], 
              t_eval: Optional[np.ndarray] = None, method: str = "rk45") -> Dict[str, Any]:
        """
        求解常微分方程组 dy/dt = f(t, y)
        
        Args:
            func: 右端函数 f(t, y)
            y0: 初始条件
            t_span: 时间区间 (t0, tf)
            t_eval: 求解时间点
            method: 求解方法
        
        Returns:
            包含解和求解信息的字典
        """
        start_time = time.time()
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)
        
        logger.info(f"求解ODE: 维度={len(y0)}, 时间步数={len(t_eval)}")
        
        if HAS_SCIPY:
            # 使用SciPy求解
            sol = integrate.solve_ivp(
                func, t_span, y0, t_eval=t_eval, method=method,
                rtol=self.config.tolerance, atol=self.config.tolerance
            )
            
            result = {
                "t": sol.t,
                "y": sol.y,
                "success": sol.success,
                "message": sol.message,
                "nfev": sol.nfev,
                "solve_time": time.time() - start_time,
                "method": method
            }
        else:
            # 简单的Runge-Kutta方法
            result = self._runge_kutta_4(func, y0, t_eval)
            result["solve_time"] = time.time() - start_time
            result["method"] = "rk4_simple"
        
        if self.config.save_history:
            self.history.append(result)
        
        logger.info(f"ODE求解完成: 方法={result['method']}, "
                   f"时间={result['solve_time']:.3f}s")
        
        return result
    
    def _runge_kutta_4(self, func: Callable, y0: np.ndarray, t_eval: np.ndarray) -> Dict[str, Any]:
        """四阶Runge-Kutta方法"""
        n = len(t_eval)
        y = np.zeros((len(y0), n))
        y[:, 0] = y0
        
        for i in range(n - 1):
            h = t_eval[i + 1] - t_eval[i]
            t = t_eval[i]
            yi = y[:, i]
            
            k1 = func(t, yi)
            k2 = func(t + h/2, yi + h*k1/2)
            k3 = func(t + h/2, yi + h*k2/2)
            k4 = func(t + h, yi + h*k3)
            
            y[:, i + 1] = yi + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return {
            "t": t_eval,
            "y": y,
            "success": True,
            "message": "Integration successful",
            "nfev": 4 * (n - 1)
        }


class PDESolver(BaseSolver):
    """偏微分方程求解器"""
    
    @performance_monitor("PDE求解")
    def solve_heat_equation(self, u0: np.ndarray, dx: float, dt: float, 
                           alpha: float, t_max: float, 
                           boundary_conditions: str = "periodic") -> Dict[str, Any]:
        """
        求解一维热方程 ∂u/∂t = α ∂²u/∂x²
        
        Args:
            u0: 初始条件
            dx: 空间步长
            dt: 时间步长
            alpha: 热扩散系数
            t_max: 最大时间
            boundary_conditions: 边界条件类型
        
        Returns:
            包含解和求解信息的字典
        """
        start_time = time.time()
        
        nx = len(u0)
        nt = int(t_max / dt) + 1
        
        # 稳定性检查
        r = alpha * dt / (dx**2)
        if r > 0.5:
            logger.warning(f"可能不稳定: r = {r:.3f} > 0.5")
        
        logger.info(f"求解热方程: 网格={nx}x{nt}, r={r:.3f}")
        
        if self.device == "cuda" and HAS_TORCH:
            result = self._solve_heat_gpu(u0, dx, dt, alpha, nt, boundary_conditions)
        else:
            result = self._solve_heat_cpu(u0, dx, dt, alpha, nt, boundary_conditions)
        
        result["solve_time"] = time.time() - start_time
        result["stability_ratio"] = r
        result["grid_size"] = (nx, nt)
        
        if self.config.save_history:
            self.history.append(result)
        
        logger.info(f"热方程求解完成: 时间={result['solve_time']:.3f}s")
        
        return result
    
    def _solve_heat_gpu(self, u0: np.ndarray, dx: float, dt: float, 
                       alpha: float, nt: int, boundary_conditions: str) -> Dict[str, Any]:
        """GPU热方程求解"""
        nx = len(u0)
        r = alpha * dt / (dx**2)
        
        # 转换到GPU
        u = torch.from_numpy(u0.astype(np.float32)).cuda()
        u_history = torch.zeros((nt, nx), device='cuda')
        u_history[0] = u
        
        for n in range(1, nt):
            u_new = u.clone()
            
            # 内部节点
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            
            # 边界条件
            if boundary_conditions == "periodic":
                u_new[0] = u[0] + r * (u[1] - 2*u[0] + u[-1])
                u_new[-1] = u[-1] + r * (u[0] - 2*u[-1] + u[-2])
            elif boundary_conditions == "neumann":
                u_new[0] = u_new[1]
                u_new[-1] = u_new[-2]
            # dirichlet边界条件保持u_new[0] = u_new[-1] = 0
            
            u = u_new
            u_history[n] = u
        
        # 转换回CPU
        u_final = self._to_numpy(u)
        u_history_np = self._to_numpy(u_history)
        
        return {
            "solution": u_final,
            "history": u_history_np,
            "method": "explicit_finite_difference_gpu",
            "boundary_conditions": boundary_conditions
        }
    
    def _solve_heat_cpu(self, u0: np.ndarray, dx: float, dt: float, 
                       alpha: float, nt: int, boundary_conditions: str) -> Dict[str, Any]:
        """CPU热方程求解"""
        nx = len(u0)
        r = alpha * dt / (dx**2)
        
        u = u0.copy()
        u_history = np.zeros((nt, nx))
        u_history[0] = u
        
        for n in range(1, nt):
            u_new = u.copy()
            
            # 内部节点
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            
            # 边界条件
            if boundary_conditions == "periodic":
                u_new[0] = u[0] + r * (u[1] - 2*u[0] + u[-1])
                u_new[-1] = u[-1] + r * (u[0] - 2*u[-1] + u[-2])
            elif boundary_conditions == "neumann":
                u_new[0] = u_new[1]
                u_new[-1] = u_new[-2]
            
            u = u_new
            u_history[n] = u
        
        return {
            "solution": u,
            "history": u_history,
            "method": "explicit_finite_difference_cpu",
            "boundary_conditions": boundary_conditions
        }


class OptimizationSolver(BaseSolver):
    """优化问题求解器"""
    
    @performance_monitor("优化求解")
    def minimize(self, func: Callable, x0: np.ndarray, method: str = "bfgs",
                bounds: Optional[List[Tuple]] = None, 
                constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        求解优化问题 min f(x)
        
        Args:
            func: 目标函数
            x0: 初始点
            method: 优化方法
            bounds: 变量界限
            constraints: 约束条件
        
        Returns:
            优化结果
        """
        start_time = time.time()
        
        logger.info(f"求解优化问题: 变量维度={len(x0)}, 方法={method}")
        
        if HAS_SCIPY:
            # 使用SciPy优化
            result = optimize.minimize(
                func, x0, method=method, bounds=bounds, constraints=constraints,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            opt_result = {
                "x": result.x,
                "fun": result.fun,
                "success": result.success,
                "message": result.message,
                "nit": result.nit,
                "nfev": result.nfev,
                "solve_time": time.time() - start_time,
                "method": method
            }
        else:
            # 简单的梯度下降
            opt_result = self._gradient_descent(func, x0)
            opt_result["solve_time"] = time.time() - start_time
            opt_result["method"] = "gradient_descent_simple"
        
        if self.config.save_history:
            self.history.append(opt_result)
        
        logger.info(f"优化完成: 函数值={opt_result['fun']:.6e}, "
                   f"迭代次数={opt_result.get('nit', 'N/A')}, "
                   f"时间={opt_result['solve_time']:.3f}s")
        
        return opt_result
    
    def _gradient_descent(self, func: Callable, x0: np.ndarray, 
                         lr: float = 0.01) -> Dict[str, Any]:
        """简单梯度下降"""
        x = x0.copy()
        
        for i in range(self.config.max_iterations):
            # 数值梯度
            grad = self._numerical_gradient(func, x)
            x_new = x - lr * grad
            
            if np.linalg.norm(x_new - x) < self.config.tolerance:
                return {
                    "x": x_new,
                    "fun": func(x_new),
                    "success": True,
                    "message": "Optimization converged",
                    "nit": i + 1,
                    "nfev": (i + 1) * (len(x) + 1)
                }
            x = x_new
        
        return {
            "x": x,
            "fun": func(x),
            "success": False,
            "message": "Maximum iterations reached",
            "nit": self.config.max_iterations,
            "nfev": self.config.max_iterations * (len(x) + 1)
        }
    
    def _numerical_gradient(self, func: Callable, x: np.ndarray, 
                           eps: float = 1e-8) -> np.ndarray:
        """数值梯度计算"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += eps
            x2[i] -= eps
            grad[i] = (func(x1) - func(x2)) / (2 * eps)
        return grad


class EigenSolver(BaseSolver):
    """特征值求解器"""
    
    @performance_monitor("特征值求解")
    def solve(self, A: np.ndarray, k: Optional[int] = None, 
             which: str = "LM", mode: str = "standard") -> Dict[str, Any]:
        """
        求解特征值问题 Ax = λx 或 Ax = λBx
        
        Args:
            A: 矩阵A
            k: 特征值数量
            which: 特征值选择策略
            mode: 求解模式
        
        Returns:
            特征值和特征向量
        """
        start_time = time.time()
        
        n = A.shape[0]
        if k is None:
            k = min(n, 6)
        
        logger.info(f"求解特征值问题: 矩阵大小={n}x{n}, 特征值数量={k}")
        
        if self.device == "cuda" and HAS_TORCH and n < 2000:
            # 小矩阵使用GPU全特征值分解
            A_tensor = self._to_device(A.astype(np.float32))
            eigenvals, eigenvecs = torch.linalg.eig(A_tensor)
            
            # 按特征值大小排序
            idx = torch.argsort(eigenvals.real, descending=True)
            eigenvals = eigenvals[idx][:k]
            eigenvecs = eigenvecs[:, idx][:, :k]
            
            eigenvals_np = self._to_numpy(eigenvals.real)
            eigenvecs_np = self._to_numpy(eigenvecs.real)
            
            result = {
                "eigenvalues": eigenvals_np,
                "eigenvectors": eigenvecs_np,
                "converged": True,
                "method": "full_decomposition_gpu"
            }
        
        elif HAS_SCIPY:
            # 使用SciPy求解
            if sparse.issparse(A) or n > 1000:
                # 大矩阵或稀疏矩阵使用稀疏求解器
                eigenvals, eigenvecs = sparse.linalg.eigs(A, k=k, which=which)
            else:
                # 密集矩阵使用密集求解器
                eigenvals, eigenvecs = linalg.eig(A)
                # 选择前k个
                idx = np.argsort(np.abs(eigenvals))[::-1]
                eigenvals = eigenvals[idx][:k]
                eigenvecs = eigenvecs[:, idx][:, :k]
            
            result = {
                "eigenvalues": eigenvals.real,
                "eigenvectors": eigenvecs.real,
                "converged": True,
                "method": "scipy_eig"
            }
        else:
            # 使用numpy
            eigenvals, eigenvecs = np.linalg.eig(A)
            idx = np.argsort(np.abs(eigenvals))[::-1]
            
            result = {
                "eigenvalues": eigenvals[idx][:k].real,
                "eigenvectors": eigenvecs[:, idx][:, :k].real,
                "converged": True,
                "method": "numpy_eig"
            }
        
        result["solve_time"] = time.time() - start_time
        result["matrix_size"] = n
        result["num_eigenvalues"] = k
        
        if self.config.save_history:
            self.history.append(result)
        
        logger.info(f"特征值求解完成: 方法={result['method']}, "
                   f"时间={result['solve_time']:.3f}s")
        
        return result


class NumericalSolverSuite:
    """数值求解器套件"""
    
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        
        # 初始化各种求解器
        self.linear_solver = LinearSolver(self.config)
        self.ode_solver = ODESolver(self.config)
        self.pde_solver = PDESolver(self.config)
        self.optimization_solver = OptimizationSolver(self.config)
        self.eigen_solver = EigenSolver(self.config)
        
        logger.info("数值求解器套件初始化完成")
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Dict[str, Any]:
        """求解线性方程组"""
        return self.linear_solver.solve(A, b, **kwargs)
    
    def solve_ode(self, func: Callable, y0: np.ndarray, t_span: Tuple[float, float], **kwargs) -> Dict[str, Any]:
        """求解常微分方程"""
        return self.ode_solver.solve(func, y0, t_span, **kwargs)
    
    def solve_heat_equation(self, u0: np.ndarray, dx: float, dt: float, alpha: float, t_max: float, **kwargs) -> Dict[str, Any]:
        """求解热方程"""
        return self.pde_solver.solve_heat_equation(u0, dx, dt, alpha, t_max, **kwargs)
    
    def minimize(self, func: Callable, x0: np.ndarray, **kwargs) -> Dict[str, Any]:
        """优化问题求解"""
        return self.optimization_solver.minimize(func, x0, **kwargs)
    
    def solve_eigenvalue(self, A: np.ndarray, **kwargs) -> Dict[str, Any]:
        """特征值问题求解"""
        return self.eigen_solver.solve(A, **kwargs)
    
    def benchmark(self) -> Dict[str, Any]:
        """性能基准测试"""
        results = {}
        
        logger.info("开始数值求解器性能基准测试...")
        
        # 线性方程组基准
        n = 1000
        A = np.random.randn(n, n)
        A = A @ A.T + np.eye(n)  # 确保正定
        b = np.random.randn(n)
        
        result = self.solve_linear_system(A, b)
        results["linear_solver"] = {
            "matrix_size": n,
            "solve_time": result["solve_time"],
            "method": result["method"],
            "converged": result["converged"]
        }
        
        # ODE基准
        def lorenz(t, y, sigma=10, rho=28, beta=8/3):
            return np.array([
                sigma * (y[1] - y[0]),
                y[0] * (rho - y[2]) - y[1],
                y[0] * y[1] - beta * y[2]
            ])
        
        y0 = np.array([1.0, 1.0, 1.0])
        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 1000)
        
        result = self.solve_ode(lorenz, y0, t_span, t_eval=t_eval)
        results["ode_solver"] = {
            "time_points": len(t_eval),
            "solve_time": result["solve_time"],
            "method": result["method"],
            "success": result["success"]
        }
        
        # PDE基准
        nx = 100
        x = np.linspace(0, 1, nx)
        u0 = np.sin(2 * np.pi * x)
        dx = 1.0 / (nx - 1)
        dt = 0.0001
        alpha = 0.01
        t_max = 0.1
        
        result = self.solve_heat_equation(u0, dx, dt, alpha, t_max)
        results["pde_solver"] = {
            "grid_size": result["grid_size"],
            "solve_time": result["solve_time"],
            "method": result["method"],
            "stability_ratio": result["stability_ratio"]
        }
        
        # 优化基准
        def rosenbrock(x):
            return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        x0 = np.array([0.0, 0.0])
        result = self.minimize(rosenbrock, x0)
        results["optimization_solver"] = {
            "variables": len(x0),
            "solve_time": result["solve_time"],
            "method": result["method"],
            "success": result["success"],
            "final_value": result["fun"]
        }
        
        # 特征值基准
        n = 500
        A = np.random.randn(n, n)
        A = A @ A.T  # 确保对称
        
        result = self.solve_eigenvalue(A, k=10)
        results["eigen_solver"] = {
            "matrix_size": n,
            "num_eigenvalues": result["num_eigenvalues"],
            "solve_time": result["solve_time"],
            "method": result["method"],
            "converged": result["converged"]
        }
        
        logger.info("数值求解器基准测试完成")
        return results


# 便捷函数
def create_solver_suite(precision: str = "float64", device: str = "auto", **kwargs) -> NumericalSolverSuite:
    """创建数值求解器套件"""
    config = SolverConfig(precision=precision, device=device, **kwargs)
    return NumericalSolverSuite(config)


def solve_linear_system(A: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
    """快速求解线性方程组"""
    solver = LinearSolver(SolverConfig(**kwargs))
    result = solver.solve(A, b)
    return result["solution"]


def solve_ode(func: Callable, y0: np.ndarray, t_span: Tuple[float, float], **kwargs) -> Dict[str, Any]:
    """快速求解ODE"""
    solver = ODESolver(SolverConfig(**kwargs))
    return solver.solve(func, y0, t_span, **kwargs) 