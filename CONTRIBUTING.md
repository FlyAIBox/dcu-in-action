# 贡献指南

感谢您对海光DCU加速卡实战项目的关注！我们欢迎所有形式的贡献，包括但不限于代码、文档、测试、反馈和建议。

## 🤝 贡献方式

### 1. 代码贡献
- 新功能开发
- Bug修复
- 性能优化
- 代码重构

### 2. 文档贡献
- 完善现有文档
- 添加新的教程
- 翻译文档
- 修正错误

### 3. 测试贡献
- 添加单元测试
- 集成测试
- 性能测试
- 兼容性测试

### 4. 社区贡献
- 回答问题
- 分享经验
- 组织活动
- 推广项目

## 📋 开发环境设置

### 前置要求
- 海光DCU设备 (BW100/K100-AI/K100/Z100L)
- Python 3.8+
- Git
- Docker (推荐)

### 环境搭建
```bash
# 1. Fork并克隆项目
git clone https://github.com/your-username/dcu-in-action.git
cd dcu-in-action

# 2. 创建开发分支
git checkout -b feature/your-feature-name

# 3. 设置开发环境
make setup

# 4. 安装依赖
make install

# 5. 运行测试
make test
```

### Docker开发环境
```bash
# 构建开发镜像
make build

# 启动开发容器
make run

# 进入容器
make shell
```

## 🔄 贡献流程

### 1. 提出Issues
在开始工作之前，请先创建或查找相关的Issue：

- **Bug报告**: 使用Bug模板详细描述问题
- **功能请求**: 说明需求和期望的解决方案
- **文档改进**: 指出需要改进的文档部分

### 2. Fork项目
1. 点击项目页面的"Fork"按钮
2. 克隆您fork的项目到本地

### 3. 创建分支
```bash
# 功能分支
git checkout -b feature/add-new-model-support

# Bug修复分支
git checkout -b fix/memory-leak-issue

# 文档分支
git checkout -b docs/update-installation-guide
```

### 4. 开发代码
- 遵循项目的代码规范
- 添加必要的测试
- 更新相关文档
- 确保所有测试通过

### 5. 提交代码
```bash
# 添加文件
git add .

# 提交更改 (遵循提交消息规范)
git commit -m "feat: add support for new model"

# 推送到远程分支
git push origin feature/add-new-model-support
```

### 6. 创建Pull Request
1. 在GitHub上创建Pull Request
2. 详细描述您的更改
3. 链接相关的Issues
4. 等待代码审查

## 📝 代码规范

### Python代码风格
我们使用以下工具来维护代码质量：

```bash
# 代码格式化
make format

# 代码检查
make lint

# 类型检查
mypy .
```

### 代码规范要求
- 使用Black进行代码格式化
- 遵循PEP 8编码规范
- 添加类型注释
- 编写清晰的文档字符串
- 保持函数和类的单一职责

### 示例代码风格
```python
from typing import Optional, List, Dict
import torch
from transformers import AutoModel

class DCUInferenceEngine:
    """DCU推理引擎基类
    
    Args:
        model_name: 模型名称或路径
        device_id: DCU设备ID
        max_length: 最大生成长度
    """
    
    def __init__(
        self, 
        model_name: str,
        device_id: int = 0,
        max_length: int = 2048
    ) -> None:
        self.model_name = model_name
        self.device_id = device_id
        self.max_length = max_length
        self._model: Optional[AutoModel] = None
    
    def load_model(self) -> None:
        """加载模型到DCU设备"""
        # 实现细节...
        pass
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 512
    ) -> str:
        """生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大新token数量
            
        Returns:
            生成的文本
        """
        # 实现细节...
        return ""
```

## 📚 文档规范

### Markdown格式
- 使用标准Markdown语法
- 中英文混排时注意空格
- 代码块指定语言类型
- 使用emoji增强可读性

### 文档结构
```markdown
# 标题

## 概述
简要介绍文档内容

## 目录
- [章节1](#章节1)
- [章节2](#章节2)

## 详细内容
### 章节1
具体内容...

### 章节2
具体内容...

## 示例代码
```python
# 示例代码
```

## 参考资源
- [相关链接](url)
```

## 🧪 测试指南

### 运行测试
```bash
# 运行所有测试
make test

# 运行特定测试
pytest tests/test_inference.py -v

# 运行DCU环境测试
make test-dcu

# 性能基准测试
make benchmark
```

### 编写测试
1. **单元测试**: 测试单个函数或类
2. **集成测试**: 测试组件间的交互
3. **性能测试**: 验证性能指标
4. **兼容性测试**: 测试不同环境的兼容性

#### 测试示例
```python
import pytest
import torch
from src.inference_engine import DCUInferenceEngine

class TestDCUInferenceEngine:
    """DCU推理引擎测试"""
    
    @pytest.fixture
    def engine(self):
        """创建测试引擎实例"""
        return DCUInferenceEngine(
            model_name="test-model",
            device_id=0
        )
    
    def test_model_loading(self, engine):
        """测试模型加载"""
        engine.load_model()
        assert engine._model is not None
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(), 
        reason="需要DCU设备"
    )
    def test_inference_on_dcu(self, engine):
        """测试DCU推理"""
        result = engine.generate("测试提示")
        assert len(result) > 0
```

## 📦 发布流程

### 版本管理
我们使用语义化版本规范：
- **主版本号**: 不兼容的API修改
- **次版本号**: 向下兼容的功能性新增
- **修订版本号**: 向下兼容的问题修正

### 发布步骤
1. 更新版本号
2. 更新CHANGELOG.md
3. 创建Release PR
4. 代码审查和测试
5. 合并到main分支
6. 创建Git标签
7. 发布Docker镜像

## 🎯 贡献优先级

### 高优先级
- **性能优化**: 提升推理和训练性能
- **Bug修复**: 修复已知问题
- **文档完善**: 补充缺失的文档
- **测试覆盖**: 增加测试用例

### 中优先级
- **新模型支持**: 增加更多模型类型
- **功能增强**: 添加新的实用功能
- **用户体验**: 改善易用性

### 低优先级
- **代码重构**: 优化代码结构
- **工具改进**: 开发辅助工具

## 🏆 贡献者奖励

### 认可机制
- **贡献者列表**: 在README中展示贡献者
- **特殊徽章**: 为不同类型的贡献颁发徽章
- **技术博客**: 邀请贡献者撰写技术文章

### 成长机会
- **代码审查**: 参与代码审查过程
- **技术讨论**: 加入核心开发团队讨论
- **会议分享**: 在技术会议上分享经验

## 📞 联系方式

### 获取帮助
- **GitHub Issues**: 技术问题和Bug报告
- **GitHub Discussions**: 功能讨论和想法交流
- **开发者社区**: https://developer.sourcefind.cn/
- **技术论坛**: https://bbs.sourcefind.cn/

### 核心维护者
- **项目负责人**: [@maintainer](https://github.com/maintainer)
- **技术负责人**: [@tech-lead](https://github.com/tech-lead)

## 📄 许可证

本项目采用MIT许可证。通过贡献代码，您同意将您的贡献按照相同许可证进行许可。

## 🙏 致谢

感谢所有为项目做出贡献的开发者！您的努力让海光DCU生态系统变得更加强大。

### 特别感谢
- 海光信息技术团队的技术支持
- 开源社区的宝贵反馈
- 所有测试用户的耐心试用

---

**让我们一起构建更好的海光DCU开发生态！** 🚀 