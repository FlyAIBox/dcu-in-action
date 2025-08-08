#!/usr/bin/env python3
"""
电力领域数据集准备脚本
用于构建和处理电力系统相关的训练数据
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
import re
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 电力领域术语表
POWER_TERMS = {
    "abbreviations": {
        "HVDC": "高压直流输电",
        "HVAC": "高压交流输电",
        "FACTS": "柔性交流输电系统",
        "SVC": "静止无功补偿器",
        "STATCOM": "静止同步补偿器",
        "DVR": "动态电压恢复器",
        "PMU": "相量测量单元",
        "SCADA": "监控与数据采集系统",
        "EMS": "能量管理系统",
        "DMS": "配电管理系统",
        "DGA": "溶解气体分析",
        "SF6": "六氟化硫",
        "GIS": "气体绝缘开关设备",
        "AIS": "空气绝缘开关设备"
    },
    "units": ["kV", "MW", "MVA", "MVar", "kA", "Hz", "Ω", "μF", "mH"],
    "keywords": ["潮流", "短路", "稳定", "保护", "变压器", "断路器", "母线", "线路", "配电", "输电"]
}


class PowerDomainDatasetBuilder:
    """电力领域数据集构建器"""
    
    def __init__(self):
        self.data_samples = []
        
    def add_qa_pair(self, question: str, answer: str, context: str = "") -> None:
        """添加问答对"""
        self.data_samples.append({
            "instruction": question,
            "input": context,
            "output": answer
        })
    
    def generate_calculation_problems(self, num_samples: int = 50) -> None:
        """生成计算类问题"""
        problem_types = [
            self._generate_power_flow_problem,
            self._generate_short_circuit_problem,
            self._generate_transformer_problem,
            self._generate_line_loss_problem,
            self._generate_reactive_power_problem
        ]
        
        for _ in range(num_samples):
            problem_func = random.choice(problem_types)
            question, answer = problem_func()
            self.add_qa_pair(question, answer)
    
    def _generate_power_flow_problem(self) -> Tuple[str, str]:
        """生成潮流计算问题"""
        voltage = random.choice([10, 35, 110, 220, 500])
        power = random.randint(10, 200)
        pf = round(random.uniform(0.8, 0.95), 2)
        
        question = f"一条{voltage}kV输电线路，输送有功功率{power}MW，功率因数为{pf}，请计算线路电流和无功功率。"
        
        current = power * 1000 / (1.732 * voltage * pf)
        reactive = power * ((1/pf**2 - 1)**0.5)
        
        answer = f"""计算过程：
1. 线路电流计算：
   I = P / (√3 × U × cosφ)
   I = {power} × 1000 / (1.732 × {voltage} × {pf})
   I = {current:.2f} A

2. 无功功率计算：
   Q = P × tan(arccos(φ))
   Q = P × √(1/cos²φ - 1)
   Q = {power} × √(1/{pf}² - 1)
   Q = {reactive:.2f} MVar

结果：
- 线路电流：{current:.2f} A
- 无功功率：{reactive:.2f} MVar"""
        
        return question, answer
    
    def _generate_short_circuit_problem(self) -> Tuple[str, str]:
        """生成短路计算问题"""
        voltage = random.choice([10.5, 35, 110])
        system_capacity = random.choice([100, 500, 1000])
        impedance = round(random.uniform(0.1, 0.5), 3)
        
        question = f"系统短路容量为{system_capacity}MVA，{voltage}kV母线通过阻抗{impedance}Ω的线路连接，计算线路末端三相短路电流。"
        
        base_current = system_capacity / (1.732 * voltage)
        system_impedance = voltage**2 / system_capacity
        line_impedance_pu = impedance / system_impedance
        total_impedance_pu = 1 + line_impedance_pu
        short_circuit_current = base_current / total_impedance_pu
        
        answer = f"""计算步骤：
1. 基准电流：Ib = Sb / (√3 × Ub) = {system_capacity} / (1.732 × {voltage}) = {base_current:.2f} kA

2. 基准阻抗：Zb = Ub² / Sb = {voltage}² / {system_capacity} = {system_impedance:.3f} Ω

3. 线路阻抗标幺值：X* = X / Zb = {impedance} / {system_impedance:.3f} = {line_impedance_pu:.3f}

4. 系统阻抗标幺值：Xs* = 1.0（基于系统短路容量）

5. 总阻抗标幺值：X∑* = Xs* + X* = 1.0 + {line_impedance_pu:.3f} = {total_impedance_pu:.3f}

6. 短路电流：Ik = Ib / X∑* = {base_current:.2f} / {total_impedance_pu:.3f} = {short_circuit_current:.2f} kA

结果：三相短路电流为 {short_circuit_current:.2f} kA"""
        
        return question, answer
    
    def _generate_transformer_problem(self) -> Tuple[str, str]:
        """生成变压器问题"""
        capacity = random.choice([31.5, 40, 50, 63, 80, 100])
        voltage_ratio = random.choice(["110/10.5", "220/110", "35/10.5"])
        load_rate = random.randint(60, 90)
        
        question = f"一台{capacity}MVA、{voltage_ratio}kV变压器，当前负载率为{load_rate}%，功率因数0.9，计算高低压侧电流。"
        
        high_v, low_v = map(float, voltage_ratio.split('/'))
        high_current = capacity * 1000 * load_rate / 100 / (1.732 * high_v)
        low_current = capacity * 1000 * load_rate / 100 / (1.732 * low_v)
        
        answer = f"""计算过程：
1. 变压器参数：
   - 额定容量：{capacity} MVA
   - 电压比：{voltage_ratio} kV
   - 负载率：{load_rate}%

2. 高压侧电流：
   I1 = S × η / (√3 × U1)
   I1 = {capacity} × 1000 × {load_rate}% / (1.732 × {high_v})
   I1 = {high_current:.2f} A

3. 低压侧电流：
   I2 = S × η / (√3 × U2)
   I2 = {capacity} × 1000 × {load_rate}% / (1.732 × {low_v})
   I2 = {low_current:.2f} A

结果：
- 高压侧电流：{high_current:.2f} A
- 低压侧电流：{low_current:.2f} A"""
        
        return question, answer
    
    def _generate_line_loss_problem(self) -> Tuple[str, str]:
        """生成线损计算问题"""
        voltage = random.choice([10, 35])
        length = random.randint(5, 50)
        current = random.randint(100, 500)
        resistance = round(random.uniform(0.1, 0.5), 2)
        
        question = f"一条{voltage}kV配电线路，长度{length}km，导线电阻{resistance}Ω/km，通过电流{current}A，计算线路损耗。"
        
        total_resistance = resistance * length
        power_loss = 3 * current**2 * total_resistance / 1000
        loss_rate = power_loss * 1000 / (1.732 * voltage * current * 0.9) * 100
        
        answer = f"""计算过程：
1. 线路总电阻：
   R = r × L = {resistance} × {length} = {total_resistance:.2f} Ω

2. 三相线路功率损耗：
   ΔP = 3 × I² × R
   ΔP = 3 × {current}² × {total_resistance:.2f}
   ΔP = {power_loss:.2f} kW

3. 线损率计算（假设功率因数0.9）：
   输送功率 P = √3 × U × I × cosφ = 1.732 × {voltage} × {current} × 0.9 = {1.732*voltage*current*0.9/1000:.2f} kW
   线损率 = ΔP / P × 100% = {power_loss:.2f} / {1.732*voltage*current*0.9/1000:.2f} × 100% = {loss_rate:.2f}%

结果：
- 线路损耗：{power_loss:.2f} kW
- 线损率：{loss_rate:.2f}%"""
        
        return question, answer
    
    def _generate_reactive_power_problem(self) -> Tuple[str, str]:
        """生成无功补偿问题"""
        power = random.randint(500, 2000)
        pf_before = round(random.uniform(0.7, 0.85), 2)
        pf_after = 0.95
        
        question = f"某工厂负荷{power}kW，功率因数为{pf_before}，需要将功率因数提高到{pf_after}，计算所需补偿容量。"
        
        tan_before = (1/pf_before**2 - 1)**0.5
        tan_after = (1/pf_after**2 - 1)**0.5
        compensation = power * (tan_before - tan_after)
        
        answer = f"""计算过程：
1. 补偿前：
   cosφ1 = {pf_before}
   tanφ1 = √(1/cos²φ1 - 1) = √(1/{pf_before}² - 1) = {tan_before:.3f}
   Q1 = P × tanφ1 = {power} × {tan_before:.3f} = {power*tan_before:.2f} kVar

2. 补偿后：
   cosφ2 = {pf_after}
   tanφ2 = √(1/cos²φ2 - 1) = √(1/{pf_after}² - 1) = {tan_after:.3f}
   Q2 = P × tanφ2 = {power} × {tan_after:.3f} = {power*tan_after:.2f} kVar

3. 补偿容量：
   Qc = Q1 - Q2 = P × (tanφ1 - tanφ2)
   Qc = {power} × ({tan_before:.3f} - {tan_after:.3f})
   Qc = {compensation:.2f} kVar

结果：需要补偿容量 {compensation:.2f} kVar"""
        
        return question, answer
    
    def add_standard_concepts(self) -> None:
        """添加标准概念解释"""
        concepts = [
            {
                "q": "请解释电力系统的静态稳定和暂态稳定的区别。",
                "a": """静态稳定和暂态稳定是电力系统稳定性的两个重要方面：

**静态稳定**：
- 定义：系统受到小扰动后，能够恢复到原始或接近原始运行状态的能力
- 特点：扰动很小，可用线性化方法分析
- 分析方法：小信号分析、特征值分析
- 判据：所有特征值实部为负
- 改善措施：励磁调节、PSS（电力系统稳定器）

**暂态稳定**：
- 定义：系统受到大扰动后，能够过渡到新的稳定运行状态的能力
- 特点：扰动较大（如短路、切机），需考虑非线性
- 分析方法：时域仿真、等面积定则
- 判据：功角不失步、电压不崩溃
- 改善措施：快速切除故障、强励磁、切机切负荷

主要区别：
1. 扰动大小：静稳是小扰动，暂稳是大扰动
2. 分析方法：静稳可线性化，暂稳必须考虑非线性
3. 时间尺度：静稳关注长期行为，暂稳关注故障后几秒内"""
            },
            {
                "q": "什么是电力系统的经济调度？包括哪些内容？",
                "a": """电力系统经济调度是在满足系统安全约束条件下，合理安排发电机组出力，使系统运行成本最小的优化过程。

**主要内容**：

1. **有功功率经济分配**：
   - 等微增率准则
   - 考虑网损的经济调度
   - 机组组合优化

2. **无功功率优化**：
   - 电压/无功优化控制
   - 无功补偿设备配置
   - 变压器分接头调整

3. **约束条件**：
   - 功率平衡约束
   - 发电机出力上下限
   - 线路潮流约束
   - 节点电压约束
   - 备用容量约束

4. **优化目标**：
   - 燃料成本最小
   - 网损最小
   - 购电成本最小
   - 环保指标（碳排放）

5. **求解方法**：
   - 经典法：拉格朗日乘数法
   - 线性规划/二次规划
   - 智能算法：遗传算法、粒子群算法

6. **实时调度**：
   - AGC（自动发电控制）
   - 5-15分钟调度周期
   - 考虑爬坡率约束"""
            }
        ]
        
        for concept in concepts:
            self.add_qa_pair(concept["q"], concept["a"])
    
    def add_troubleshooting_cases(self) -> None:
        """添加故障处理案例"""
        cases = [
            {
                "q": "220kV变电站主变压器油色谱分析显示乙炔含量达到5ppm，请分析可能的故障原因和处理建议。",
                "a": """乙炔(C2H2)是变压器内部高能放电的特征气体，5ppm已超过注意值(1-2ppm)，需要高度重视。

**可能的故障原因**：
1. 内部电弧放电
2. 分接开关接触不良产生电弧
3. 引线连接松动放电
4. 绕组匝间或层间短路

**进一步诊断**：
1. 进行三比值分析，确定故障类型
2. 测量其他特征气体(H2、CH4、C2H4等)
3. 进行电气试验(绝缘电阻、介损、直阻等)
4. 必要时进行局部放电测试

**处理建议**：
1. 立即加强监测，缩短取样周期至每周
2. 限制变压器负载至80%以下
3. 准备备用变压器或制定转供方案
4. 如乙炔继续增长，择机停电检查
5. 考虑进行吊罩检查或返厂检修

**预防措施**：
- 定期进行预防性试验
- 加强运行巡视
- 控制负载和温度
- 完善在线监测系统"""
            }
        ]
        
        for case in cases:
            self.add_qa_pair(case["q"], case["a"])
    
    def generate_from_standards(self, standards_file: str = None) -> None:
        """从电力标准规范生成问答"""
        # 这里可以解析电力行业标准文档
        # 示例：DL/T、GB/T等标准
        pass
    
    def save_dataset(self, output_path: str) -> None:
        """保存数据集"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存了 {len(self.data_samples)} 条数据到 {output_file}")


def create_dataset_info(dataset_name: str, output_dir: Path) -> None:
    """创建dataset_info.json"""
    dataset_info = {
        dataset_name: {
            "file_name": f"{dataset_name}.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input", 
                "response": "output"
            }
        },
        f"{dataset_name}_samples": {
            "file_name": "power_domain_samples.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }
    
    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"创建数据集配置文件: {info_path}")


def merge_external_data(external_file: str, builder: PowerDomainDatasetBuilder) -> None:
    """合并外部数据"""
    file_path = Path(external_file)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        logger.warning(f"不支持的文件格式: {file_path.suffix}")
        return
    
    for item in data:
        if 'instruction' in item and 'output' in item:
            builder.add_qa_pair(
                item['instruction'],
                item['output'],
                item.get('input', '')
            )


def main():
    parser = argparse.ArgumentParser(description="构建电力领域专用数据集")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="输出目录")
    parser.add_argument("--dataset_name", type=str, default="power_domain",
                       help="数据集名称")
    parser.add_argument("--num_calc_problems", type=int, default=100,
                       help="生成计算题数量")
    parser.add_argument("--external_data", type=str, nargs='*',
                       help="外部数据文件列表")
    parser.add_argument("--include_samples", action='store_true',
                       help="包含示例数据")
    
    args = parser.parse_args()
    
    # 创建数据集构建器
    builder = PowerDomainDatasetBuilder()
    
    logger.info("开始构建电力领域数据集...")
    
    # 添加标准概念
    logger.info("添加标准概念解释...")
    builder.add_standard_concepts()
    
    # 添加故障案例
    logger.info("添加故障处理案例...")
    builder.add_troubleshooting_cases()
    
    # 生成计算题
    logger.info(f"生成 {args.num_calc_problems} 道计算题...")
    builder.generate_calculation_problems(args.num_calc_problems)
    
    # 合并外部数据
    if args.external_data:
        for ext_file in args.external_data:
            logger.info(f"合并外部数据: {ext_file}")
            merge_external_data(ext_file, builder)
    
    # 如果需要包含示例数据
    if args.include_samples:
        samples_file = Path(args.output_dir) / "power_domain_samples.json"
        if samples_file.exists():
            logger.info("合并示例数据...")
            merge_external_data(str(samples_file), builder)
    
    # 保存数据集
    output_path = Path(args.output_dir) / f"{args.dataset_name}.json"
    builder.save_dataset(output_path)
    
    # 创建配置文件
    create_dataset_info(args.dataset_name, Path(args.output_dir))
    
    # 显示统计信息
    logger.info(f"\n数据集统计:")
    logger.info(f"总样本数: {len(builder.data_samples)}")
    logger.info(f"保存位置: {output_path}")
    
    # 显示样例
    if builder.data_samples:
        logger.info("\n数据样例:")
        sample = builder.data_samples[0]
        print(f"问题: {sample['instruction'][:100]}...")
        if sample['input']:
            print(f"上下文: {sample['input'][:100]}...")
        print(f"答案: {sample['output'][:100]}...")


if __name__ == "__main__":
    main() 