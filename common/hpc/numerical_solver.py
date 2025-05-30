"""
数值求解器

提供科学计算中常用的数值方法和求解器。
"""

import logging
import numpy as np
import torch
from typing import Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..dcu import DCUManager


@dataclass
class SolverConfig:
    """求解器配置"""
    tolerance: float = 1e-6
    max_iterations: int = 1000
    device: str = "dcu"
    dtype: str = "float32"


class NumericalSolver:
    """数值求解器"""
    
    def __init__(self, config: Optional[SolverConfig] = None, dcu_manager: Optional[DCUManager] = None):
        """
        初始化数值求解器
        
        Args:
            config: 求解器配置
            dcu_manager: DCU管理器
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or SolverConfig()
        self.dcu_manager = dcu_manager or DCUManager()
        
        # 设置计算设备
        if self.config.device == "dcu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 设置数据类型
        if self.config.dtype == "float32":
            self.dtype = torch.float32
        elif self.config.dtype == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
    
    def newton_raphson(self, 
                      func: Callable[[torch.Tensor], torch.Tensor],
                      derivative: Callable[[torch.Tensor], torch.Tensor],
                      initial_guess: float,
                      tolerance: Optional[float] = None,
                      max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        牛顿-拉夫逊法求解非线性方程
        
        Args:
            func: 目标函数
            derivative: 导数函数
            initial_guess: 初始猜测值
            tolerance: 容差
            max_iterations: 最大迭代次数
            
        Returns:
            求解结果字典
        """
        tolerance = tolerance or self.config.tolerance
        max_iterations = max_iterations or self.config.max_iterations
        
        x = torch.tensor(initial_guess, dtype=self.dtype, device=self.device)
        
        for i in range(max_iterations):
            fx = func(x)
            dfx = derivative(x)
            
            if torch.abs(dfx) < 1e-12:
                self.logger.warning("导数接近零，可能导致数值不稳定")
                break
            
            x_new = x - fx / dfx
            
            if torch.abs(x_new - x) < tolerance:
                return {
                    "solution": x_new.item(),
                    "iterations": i + 1,
                    "converged": True,
                    "final_error": torch.abs(x_new - x).item()
                }
            
            x = x_new
        
        return {
            "solution": x.item(),
            "iterations": max_iterations,
            "converged": False,
            "final_error": torch.abs(func(x)).item()
        }
    
    def gradient_descent(self,
                        objective: Callable[[torch.Tensor], torch.Tensor],
                        gradient: Callable[[torch.Tensor], torch.Tensor],
                        initial_point: torch.Tensor,
                        learning_rate: float = 0.01,
                        tolerance: Optional[float] = None,
                        max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        梯度下降优化
        
        Args:
            objective: 目标函数
            gradient: 梯度函数
            initial_point: 初始点
            learning_rate: 学习率
            tolerance: 容差
            max_iterations: 最大迭代次数
            
        Returns:
            优化结果字典
        """
        tolerance = tolerance or self.config.tolerance
        max_iterations = max_iterations or self.config.max_iterations
        
        x = initial_point.to(device=self.device, dtype=self.dtype)
        history = []
        
        for i in range(max_iterations):
            grad = gradient(x)
            x_new = x - learning_rate * grad
            
            # 记录历史
            obj_value = objective(x).item()
            history.append({
                "iteration": i,
                "x": x.clone().detach(),
                "objective": obj_value,
                "gradient_norm": torch.norm(grad).item()
            })
            
            if torch.norm(grad) < tolerance:
                return {
                    "solution": x_new,
                    "iterations": i + 1,
                    "converged": True,
                    "final_objective": objective(x_new).item(),
                    "history": history
                }
            
            x = x_new
        
        return {
            "solution": x,
            "iterations": max_iterations,
            "converged": False,
            "final_objective": objective(x).item(),
            "history": history
        }
    
    def runge_kutta_4(self,
                     func: Callable[[float, torch.Tensor], torch.Tensor],
                     t_span: Tuple[float, float],
                     y0: torch.Tensor,
                     num_points: int = 100) -> Dict[str, torch.Tensor]:
        """
        四阶龙格-库塔法求解常微分方程
        
        Args:
            func: 右端函数 dy/dt = f(t, y)
            t_span: 时间区间 (t0, tf)
            y0: 初始条件
            num_points: 时间点数量
            
        Returns:
            求解结果
        """
        t0, tf = t_span
        h = (tf - t0) / (num_points - 1)
        
        t = torch.linspace(t0, tf, num_points, dtype=self.dtype, device=self.device)
        y = torch.zeros((num_points, *y0.shape), dtype=self.dtype, device=self.device)
        y[0] = y0.to(device=self.device, dtype=self.dtype)
        
        for i in range(num_points - 1):
            ti = t[i].item()
            yi = y[i]
            
            k1 = func(ti, yi)
            k2 = func(ti + h/2, yi + h*k1/2)
            k3 = func(ti + h/2, yi + h*k2/2)
            k4 = func(ti + h, yi + h*k3)
            
            y[i+1] = yi + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return {
            "t": t,
            "y": y,
            "success": True
        }
    
    def bisection_method(self,
                        func: Callable[[torch.Tensor], torch.Tensor],
                        interval: Tuple[float, float],
                        tolerance: Optional[float] = None,
                        max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        二分法求解方程根
        
        Args:
            func: 目标函数
            interval: 搜索区间
            tolerance: 容差
            max_iterations: 最大迭代次数
            
        Returns:
            求解结果
        """
        tolerance = tolerance or self.config.tolerance
        max_iterations = max_iterations or self.config.max_iterations
        
        a, b = interval
        a = torch.tensor(a, dtype=self.dtype, device=self.device)
        b = torch.tensor(b, dtype=self.dtype, device=self.device)
        
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            return {
                "solution": None,
                "iterations": 0,
                "converged": False,
                "error": "函数在区间端点同号，无法使用二分法"
            }
        
        for i in range(max_iterations):
            c = (a + b) / 2
            fc = func(c)
            
            if torch.abs(fc) < tolerance or torch.abs(b - a) < tolerance:
                return {
                    "solution": c.item(),
                    "iterations": i + 1,
                    "converged": True,
                    "final_error": torch.abs(fc).item()
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return {
            "solution": ((a + b) / 2).item(),
            "iterations": max_iterations,
            "converged": False,
            "final_error": torch.abs(func((a + b) / 2)).item()
        }
    
    def monte_carlo_integration(self,
                               func: Callable[[torch.Tensor], torch.Tensor],
                               domain: Tuple[Tuple[float, float], ...],
                               num_samples: int = 1000000) -> Dict[str, Any]:
        """
        蒙特卡罗积分
        
        Args:
            func: 被积函数
            domain: 积分域（每个维度的范围）
            num_samples: 采样点数
            
        Returns:
            积分结果
        """
        dim = len(domain)
        
        # 生成随机采样点
        samples = torch.zeros((num_samples, dim), dtype=self.dtype, device=self.device)
        volume = 1.0
        
        for i, (a, b) in enumerate(domain):
            samples[:, i] = torch.rand(num_samples, dtype=self.dtype, device=self.device) * (b - a) + a
            volume *= (b - a)
        
        # 计算函数值
        values = torch.zeros(num_samples, dtype=self.dtype, device=self.device)
        batch_size = min(10000, num_samples)  # 分批处理避免内存溢出
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_samples = samples[i:end_idx]
            batch_values = torch.stack([func(sample) for sample in batch_samples])
            values[i:end_idx] = batch_values.squeeze()
        
        # 计算积分值和标准误差
        integral = torch.mean(values) * volume
        std_error = torch.std(values) * volume / torch.sqrt(torch.tensor(num_samples, dtype=self.dtype))
        
        return {
            "integral": integral.item(),
            "standard_error": std_error.item(),
            "num_samples": num_samples,
            "domain_volume": volume
        }
    
    def get_solver_info(self) -> Dict[str, Any]:
        """获取求解器信息"""
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "config": {
                "tolerance": self.config.tolerance,
                "max_iterations": self.config.max_iterations,
                "device": self.config.device,
                "dtype": self.config.dtype
            }
        } 