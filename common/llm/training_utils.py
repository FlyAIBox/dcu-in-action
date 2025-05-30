"""
大模型训练工具库
提供分布式训练、混合精度、检查点管理、优化器配置等训练相关功能
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, asdict
import math
import shutil
from contextlib import contextmanager
import gc

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        Trainer, TrainingArguments
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    from apex import amp
    HAS_APEX = True
except ImportError:
    HAS_APEX = False

from ..utils.logger import get_logger, performance_monitor
from ..utils.monitor import SystemMonitor

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name_or_path: str
    model_type: str = "auto"
    
    # 训练参数
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 优化器配置
    optimizer_type: str = "adamw"  # adamw, sgd, adafactor
    scheduler_type: str = "linear"  # linear, cosine, constant
    
    # 混合精度
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    
    # 分布式训练
    distributed: bool = False
    local_rank: int = -1
    deepspeed_config: Optional[str] = None
    
    # 数据配置
    max_seq_length: int = 512
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # 保存和日志
    output_dir: str = "./output"
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # 其他
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    gradient_accumulation_steps: int = 1


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    step: int
    learning_rate: float
    loss: float
    grad_norm: float
    throughput: float  # samples/sec
    memory_usage: float  # GB
    timestamp: str


class TrainingState:
    """训练状态管理"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = time.time()
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def save(self, filepath: str):
        """保存训练状态"""
        state = {
            'config': asdict(self.config),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'start_time': self.start_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, filepath: str):
        """加载训练状态"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.global_step = state['global_step']
        self.epoch = state['epoch']
        self.best_loss = state['best_loss']
        self.start_time = state['start_time']
        
        self.metrics_history = [
            TrainingMetrics(**m) for m in state['metrics_history']
        ]


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, 
                 save_dir: str,
                 max_checkpoints: int = 3,
                 save_optimizer: bool = True,
                 save_scheduler: bool = True):
        
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any] = None,
                       epoch: int = 0,
                       step: int = 0,
                       loss: float = 0.0,
                       metrics: Optional[Dict] = None):
        """保存检查点"""
        
        checkpoint_name = f"checkpoint-epoch-{epoch}-step-{step}"
        checkpoint_dir = self.save_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存模型
        if hasattr(model, 'module'):
            # DDP模型
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        torch.save(model_state, checkpoint_dir / "pytorch_model.bin")
        
        # 保存优化器
        if self.save_optimizer and optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # 保存调度器
        if self.save_scheduler and scheduler is not None:
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        # 保存训练信息
        train_info = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'metrics': metrics or {},
            'timestamp': time.time()
        }
        
        with open(checkpoint_dir / "training_args.json", 'w') as f:
            json.dump(train_info, f, indent=2)
        
        # 管理检查点数量
        self.checkpoints.append((checkpoint_dir, loss))
        self.checkpoints.sort(key=lambda x: x[1])  # 按loss排序
        
        if len(self.checkpoints) > self.max_checkpoints:
            # 删除最差的检查点
            worst_checkpoint, _ = self.checkpoints.pop()
            if worst_checkpoint.exists():
                shutil.rmtree(worst_checkpoint)
                logger.info(f"删除检查点: {worst_checkpoint}")
        
        logger.info(f"保存检查点: {checkpoint_dir}")
        return checkpoint_dir
    
    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """加载检查点"""
        
        if checkpoint_path is None:
            # 找最新的检查点
            checkpoints = list(self.save_dir.glob("checkpoint-*"))
            if not checkpoints:
                logger.warning("未找到检查点")
                return {}
            
            checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        checkpoint_path = Path(checkpoint_path)
        
        # 加载模型
        model_file = checkpoint_path / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location='cpu')
            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            logger.info(f"加载模型: {model_file}")
        
        # 加载优化器
        if optimizer is not None:
            optimizer_file = checkpoint_path / "optimizer.pt"
            if optimizer_file.exists():
                optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))
                logger.info(f"加载优化器: {optimizer_file}")
        
        # 加载调度器
        if scheduler is not None:
            scheduler_file = checkpoint_path / "scheduler.pt"
            if scheduler_file.exists():
                scheduler.load_state_dict(torch.load(scheduler_file, map_location='cpu'))
                logger.info(f"加载调度器: {scheduler_file}")
        
        # 加载训练信息
        info_file = checkpoint_path / "training_args.json"
        train_info = {}
        if info_file.exists():
            with open(info_file, 'r') as f:
                train_info = json.load(f)
        
        logger.info(f"从检查点恢复: {checkpoint_path}")
        return train_info
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """获取最佳检查点"""
        if not self.checkpoints:
            return None
        return self.checkpoints[0][0]  # 最小loss的检查点


class DistributedTrainingManager:
    """分布式训练管理器"""
    
    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.is_initialized = False
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
    
    def setup(self):
        """初始化分布式训练"""
        if self.world_size > 1 and not self.is_initialized:
            dist.init_process_group(backend=self.backend)
            self.is_initialized = True
            logger.info(f"分布式训练初始化完成: rank={self.rank}, world_size={self.world_size}")
    
    def cleanup(self):
        """清理分布式训练"""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
    
    def is_main_process(self) -> bool:
        """是否为主进程"""
        return self.rank == 0
    
    def barrier(self):
        """同步所有进程"""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """所有进程的张量求和"""
        if self.is_initialized:
            dist.all_reduce(tensor, op)
        return tensor


class MixedPrecisionManager:
    """混合精度训练管理器"""
    
    def __init__(self, 
                 enabled: bool = False,
                 use_bf16: bool = False,
                 loss_scale: Optional[float] = None):
        
        self.enabled = enabled
        self.use_bf16 = use_bf16
        self.scaler = None
        
        if enabled:
            if use_bf16:
                # BF16不需要GradScaler
                self.autocast_dtype = torch.bfloat16
            else:
                # FP16需要GradScaler
                self.autocast_dtype = torch.float16
                self.scaler = GradScaler(init_scale=loss_scale or 2**16)
    
    @contextmanager
    def autocast_context(self):
        """自动混合精度上下文"""
        if self.enabled:
            with autocast(dtype=self.autocast_dtype):
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失（仅FP16）"""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """优化器步进"""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor):
        """反向传播"""
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


class OptimizerFactory:
    """优化器工厂"""
    
    @staticmethod
    def create_optimizer(model: nn.Module,
                        optimizer_type: str = "adamw",
                        learning_rate: float = 5e-5,
                        weight_decay: float = 0.01,
                        **kwargs) -> torch.optim.Optimizer:
        """创建优化器"""
        
        # 获取参数组
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8)
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=learning_rate,
                momentum=kwargs.get("momentum", 0.9)
            )
        elif optimizer_type.lower() == "adafactor" and HAS_TRANSFORMERS:
            from transformers import Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=learning_rate,
                scale_parameter=kwargs.get("scale_parameter", True),
                relative_step_size=kwargs.get("relative_step_size", False)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer,
                        scheduler_type: str = "linear",
                        num_training_steps: int = 1000,
                        num_warmup_steps: int = 100,
                        **kwargs):
        """创建学习率调度器"""
        
        if scheduler_type.lower() == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type.lower() == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=kwargs.get("num_cycles", 0.5)
            )
        elif scheduler_type.lower() == "constant":
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
        
        return scheduler


class TrainingLoop:
    """训练循环管理器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState(config)
        
        # 初始化各种管理器
        self.dist_manager = DistributedTrainingManager()
        self.mp_manager = MixedPrecisionManager(
            enabled=config.fp16 or config.bf16,
            use_bf16=config.bf16
        )
        self.checkpoint_manager = CheckpointManager(
            save_dir=os.path.join(config.output_dir, "checkpoints"),
            max_checkpoints=config.save_total_limit
        )
        
        # 监控
        self.monitor = SystemMonitor(interval=10.0)
        
        # 设置设备
        self.device = self._setup_device()
        
        # 设置随机种子
        self._set_seed(config.seed)
    
    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if torch.cuda.is_available():
            if self.dist_manager.local_rank != -1:
                device = torch.device(f"cuda:{self.dist_manager.local_rank}")
                torch.cuda.set_device(device)
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        logger.info(f"使用设备: {device}")
        return device
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @performance_monitor("模型初始化")
    def setup_model(self, model: nn.Module) -> nn.Module:
        """设置模型"""
        model = model.to(self.device)
        
        # 梯度检查点
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            else:
                logger.warning("模型不支持梯度检查点")
        
        # 分布式训练
        if self.dist_manager.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.dist_manager.local_rank],
                output_device=self.dist_manager.local_rank,
                find_unused_parameters=False
            )
        
        return model
    
    def setup_training_components(self, 
                                model: nn.Module,
                                train_dataloader: DataLoader,
                                eval_dataloader: Optional[DataLoader] = None):
        """设置训练组件"""
        
        # 计算训练步数
        total_steps = len(train_dataloader) * self.config.num_epochs
        total_steps = total_steps // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # 创建优化器和调度器
        optimizer = OptimizerFactory.create_optimizer(
            model,
            optimizer_type=self.config.optimizer_type,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = OptimizerFactory.create_scheduler(
            optimizer,
            scheduler_type=self.config.scheduler_type,
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps
        )
        
        return optimizer, scheduler, total_steps
    
    def train_epoch(self,
                   model: nn.Module,
                   train_dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   scheduler,
                   epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        
        model.train()
        total_loss = 0.0
        num_steps = 0
        
        # 开始监控
        if epoch == 0:
            self.monitor.start()
        
        for step, batch in enumerate(train_dataloader):
            # 移动数据到设备
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # 前向传播
            with self.mp_manager.autocast_context():
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # 梯度累积
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            self.mp_manager.backward(loss)
            
            # 梯度累积步骤
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.max_grad_norm > 0:
                    if hasattr(model, 'module'):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.module.parameters(), 
                            self.config.max_grad_norm
                        )
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            self.config.max_grad_norm
                        )
                else:
                    grad_norm = 0.0
                
                # 优化器步进
                self.mp_manager.step_optimizer(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新状态
                self.state.global_step += 1
                num_steps += 1
                
                # 记录指标
                if self.state.global_step % self.config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # 计算吞吐量
                    elapsed = time.time() - self.state.start_time
                    throughput = (self.state.global_step * self.config.batch_size) / elapsed
                    
                    # 获取显存使用
                    memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    
                    # 记录指标
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        step=self.state.global_step,
                        learning_rate=current_lr,
                        loss=loss.item(),
                        grad_norm=float(grad_norm),
                        throughput=throughput,
                        memory_usage=memory_usage,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    self.state.metrics_history.append(metrics)
                    
                    if self.dist_manager.is_main_process():
                        logger.info(
                            f"Epoch {epoch}, Step {self.state.global_step}: "
                            f"loss={loss.item():.4f}, lr={current_lr:.2e}, "
                            f"throughput={throughput:.1f} samples/s"
                        )
                
                # 保存检查点
                if (self.config.save_steps > 0 and 
                    self.state.global_step % self.config.save_steps == 0 and
                    self.dist_manager.is_main_process()):
                    
                    self.checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=self.state.global_step,
                        loss=loss.item()
                    )
            
            total_loss += loss.item()
        
        return {
            "train_loss": total_loss / max(num_steps, 1),
            "learning_rate": scheduler.get_last_lr()[0] if scheduler else 0
        }
    
    def evaluate(self, 
                model: nn.Module, 
                eval_dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        total_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # 移动数据到设备
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # 前向传播
                with self.mp_manager.autocast_context():
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
                num_steps += 1
        
        eval_loss = total_loss / max(num_steps, 1)
        
        # 分布式评估时聚合结果
        if self.dist_manager.world_size > 1:
            eval_loss_tensor = torch.tensor(eval_loss, device=self.device)
            self.dist_manager.all_reduce(eval_loss_tensor)
            eval_loss = eval_loss_tensor.item() / self.dist_manager.world_size
        
        return {"eval_loss": eval_loss}
    
    def train(self,
             model: nn.Module,
             train_dataloader: DataLoader,
             eval_dataloader: Optional[DataLoader] = None):
        """完整训练流程"""
        
        # 初始化分布式训练
        self.dist_manager.setup()
        
        try:
            # 设置模型
            model = self.setup_model(model)
            
            # 设置训练组件
            optimizer, scheduler, total_steps = self.setup_training_components(
                model, train_dataloader, eval_dataloader
            )
            
            # 恢复检查点
            if self.config.resume_from_checkpoint:
                checkpoint_info = self.checkpoint_manager.load_checkpoint(
                    model, optimizer, scheduler, self.config.resume_from_checkpoint
                )
                if checkpoint_info:
                    self.state.epoch = checkpoint_info.get('epoch', 0)
                    self.state.global_step = checkpoint_info.get('step', 0)
            
            logger.info(f"开始训练: {self.config.num_epochs} epochs, {total_steps} steps")
            
            # 训练循环
            for epoch in range(self.state.epoch, self.config.num_epochs):
                self.state.epoch = epoch
                
                # 设置分布式采样器的epoch
                if hasattr(train_dataloader.sampler, 'set_epoch'):
                    train_dataloader.sampler.set_epoch(epoch)
                
                # 训练一个epoch
                train_metrics = self.train_epoch(
                    model, train_dataloader, optimizer, scheduler, epoch
                )
                
                # 评估
                eval_metrics = {}
                if eval_dataloader is not None and self.config.eval_steps > 0:
                    if (epoch + 1) % (self.config.eval_steps // len(train_dataloader)) == 0:
                        eval_metrics = self.evaluate(model, eval_dataloader)
                
                # 记录epoch结果
                if self.dist_manager.is_main_process():
                    epoch_metrics = {**train_metrics, **eval_metrics}
                    logger.info(f"Epoch {epoch} 完成: {epoch_metrics}")
                    
                    # 保存最佳模型
                    current_loss = eval_metrics.get('eval_loss', train_metrics['train_loss'])
                    if current_loss < self.state.best_loss:
                        self.state.best_loss = current_loss
                        
                        # 保存最佳检查点
                        best_checkpoint = self.checkpoint_manager.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            step=self.state.global_step,
                            loss=current_loss,
                            metrics=epoch_metrics
                        )
                        
                        # 保存最终模型
                        final_model_dir = Path(self.config.output_dir) / "best_model"
                        final_model_dir.mkdir(exist_ok=True)
                        
                        if hasattr(model, 'module'):
                            model.module.save_pretrained(final_model_dir)
                        elif hasattr(model, 'save_pretrained'):
                            model.save_pretrained(final_model_dir)
                        else:
                            torch.save(model.state_dict(), final_model_dir / "pytorch_model.bin")
                
                # 内存清理
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("训练完成")
            
            # 保存训练状态
            if self.dist_manager.is_main_process():
                self.state.save(os.path.join(self.config.output_dir, "training_state.json"))
                
                # 保存监控报告
                self.monitor.save_report(
                    os.path.join(self.config.output_dir, "training_monitor.json")
                )
        
        finally:
            # 清理
            self.monitor.stop()
            self.dist_manager.cleanup()


def create_distributed_dataloader(dataset,
                                 batch_size: int,
                                 shuffle: bool = True,
                                 num_workers: int = 4,
                                 pin_memory: bool = True) -> DataLoader:
    """创建分布式数据加载器"""
    
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # sampler已处理shuffle
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    return dataloader


def setup_deepspeed_training(model: nn.Module,
                           config_path: str,
                           training_args: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """设置DeepSpeed训练"""
    
    if not HAS_DEEPSPEED:
        raise ImportError("需要安装DeepSpeed: pip install deepspeed")
    
    # 读取DeepSpeed配置
    with open(config_path, 'r') as f:
        ds_config = json.load(f)
    
    # 更新配置
    ds_config.update({
        "train_batch_size": training_args.get("train_batch_size", 16),
        "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps", 1),
        "steps_per_print": training_args.get("logging_steps", 10),
    })
    
    # 初始化DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    return model_engine, optimizer, lr_scheduler, ds_config


# 预设训练配置
PRESET_CONFIGS = {
    "small_model_fp16": TrainingConfig(
        model_name_or_path="",
        batch_size=16,
        learning_rate=5e-5,
        num_epochs=3,
        fp16=True,
        gradient_accumulation_steps=1,
        max_seq_length=512
    ),
    
    "large_model_distributed": TrainingConfig(
        model_name_or_path="",
        batch_size=4,
        learning_rate=1e-5,
        num_epochs=1,
        fp16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        distributed=True,
        max_seq_length=2048
    ),
    
    "debug_mode": TrainingConfig(
        model_name_or_path="",
        batch_size=2,
        learning_rate=1e-4,
        num_epochs=1,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        max_seq_length=128
    )
}


def get_preset_config(preset_name: str, **overrides) -> TrainingConfig:
    """获取预设配置"""
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"未知的预设配置: {preset_name}。可用配置: {available}")
    
    config = PRESET_CONFIGS[preset_name]
    
    # 应用覆盖参数
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"忽略未知的配置参数: {key}")
    
    return config 