"""
HPC高性能计算模块
提供科学计算、数值求解、并行计算等功能
"""

from .numerical_solver import (
    SolverConfig,
    BaseSolver,
    LinearSolver,
    ODESolver,
    PDESolver,
    OptimizationSolver,
    EigenSolver,
    NumericalSolverSuite,
    create_solver_suite,
    solve_linear_system,
    solve_ode
)

__all__ = [
    "SolverConfig",
    "BaseSolver",
    "LinearSolver",
    "ODESolver", 
    "PDESolver",
    "OptimizationSolver",
    "EigenSolver",
    "NumericalSolverSuite",
    "create_solver_suite",
    "solve_linear_system",
    "solve_ode",
] 