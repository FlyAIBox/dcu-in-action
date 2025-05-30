# DCU-in-Action Makefile
# 提供完整的项目管理和部署功能

.PHONY: help install test clean build deploy docker-build docker-run benchmark docs lint format

# 默认目标
.DEFAULT_GOAL := help

# 项目配置
PROJECT_NAME := dcu-in-action
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := $(PROJECT_NAME):latest
DOCKER_REGISTRY := your-registry.com

# 颜色定义
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## 显示帮助信息
	@echo "$(BLUE)DCU-in-Action 项目管理工具$(RESET)"
	@echo "$(BLUE)================================$(RESET)"
	@echo ""
	@echo "$(GREEN)可用命令:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)使用示例:$(RESET)"
	@echo "  make install     # 安装项目依赖"
	@echo "  make test        # 运行测试"
	@echo "  make benchmark   # 运行性能基准测试"
	@echo "  make docker-run  # 启动Docker开发环境"

install: ## 安装项目依赖和环境
	@echo "$(GREEN)🚀 安装DCU-in-Action环境...$(RESET)"
	@echo "$(BLUE)检查DCU环境...$(RESET)"
	@bash common/setup/check_environment.sh
	@echo "$(BLUE)安装Python依赖...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(BLUE)安装项目...$(RESET)"
	$(PIP) install -e .
	@echo "$(BLUE)设置权限...$(RESET)"
	chmod +x common/setup/*.sh
	chmod +x scripts/**/*.sh
	@echo "$(GREEN)✅ 安装完成！$(RESET)"

install-dev: ## 安装开发环境依赖
	@echo "$(GREEN)🛠️ 安装开发环境...$(RESET)"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install pre-commit black flake8 pytest pytest-cov
	pre-commit install
	@echo "$(GREEN)✅ 开发环境安装完成！$(RESET)"

test: ## 运行测试套件
	@echo "$(GREEN)🧪 运行测试套件...$(RESET)"
	@echo "$(BLUE)基础环境测试...$(RESET)"
	$(PYTHON) examples/basic/hello_dcu.py
	@echo "$(BLUE)单元测试...$(RESET)"
	pytest tests/ -v --cov=common --cov-report=html
	@echo "$(GREEN)✅ 测试完成！$(RESET)"

test-quick: ## 快速测试（跳过耗时测试）
	@echo "$(GREEN)⚡ 快速测试...$(RESET)"
	$(PYTHON) examples/basic/hello_dcu.py
	pytest tests/ -v -m "not slow"
	@echo "$(GREEN)✅ 快速测试完成！$(RESET)"

benchmark: ## 运行性能基准测试
	@echo "$(GREEN)📊 运行性能基准测试...$(RESET)"
	$(PYTHON) examples/benchmarks/dcu_benchmark.py
	@echo "$(GREEN)✅ 基准测试完成！$(RESET)"

benchmark-full: ## 运行完整性能基准测试
	@echo "$(GREEN)📊 运行完整性能基准测试...$(RESET)"
	$(PYTHON) examples/benchmarks/dcu_benchmark.py --profile
	@echo "$(GREEN)✅ 完整基准测试完成！$(RESET)"

clean: ## 清理临时文件和缓存
	@echo "$(GREEN)🧹 清理项目...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ .coverage htmlcov/
	rm -rf .pytest_cache/ .tox/
	@echo "$(GREEN)✅ 清理完成！$(RESET)"

format: ## 格式化代码
	@echo "$(GREEN)🎨 格式化代码...$(RESET)"
	black common/ examples/ tests/
	isort common/ examples/ tests/
	@echo "$(GREEN)✅ 代码格式化完成！$(RESET)"

lint: ## 代码质量检查
	@echo "$(GREEN)🔍 代码质量检查...$(RESET)"
	flake8 common/ examples/ tests/
	black --check common/ examples/ tests/
	isort --check-only common/ examples/ tests/
	@echo "$(GREEN)✅ 代码检查完成！$(RESET)"

docs: ## 生成文档
	@echo "$(GREEN)📚 生成文档...$(RESET)"
	cd docs && make html
	@echo "$(GREEN)✅ 文档生成完成！$(RESET)"

docs-serve: ## 启动文档服务器
	@echo "$(GREEN)📚 启动文档服务器...$(RESET)"
	cd docs/_build/html && python -m http.server 8080
	@echo "$(BLUE)文档服务器运行在: http://localhost:8080$(RESET)"

# Docker相关命令
docker-build: ## 构建Docker镜像
	@echo "$(GREEN)🐳 构建Docker镜像...$(RESET)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✅ Docker镜像构建完成！$(RESET)"

docker-run: ## 运行Docker开发环境
	@echo "$(GREEN)🐳 启动Docker开发环境...$(RESET)"
	docker run -it --rm \
		--device=/dev/kfd \
		--device=/dev/dri \
		-v $(PWD):/workspace \
		-p 8888:8888 \
		-p 6006:6006 \
		-p 8000:8000 \
		$(DOCKER_IMAGE)

docker-compose-up: ## 启动完整Docker Compose环境
	@echo "$(GREEN)🐳 启动Docker Compose环境...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)✅ Docker Compose环境启动完成！$(RESET)"
	@echo "$(BLUE)服务访问地址:$(RESET)"
	@echo "  - Jupyter: http://localhost:8888"
	@echo "  - TensorBoard: http://localhost:6006"
	@echo "  - API服务: http://localhost:8000"
	@echo "  - Grafana: http://localhost:3000"

docker-compose-down: ## 停止Docker Compose环境
	@echo "$(GREEN)🐳 停止Docker Compose环境...$(RESET)"
	docker-compose down
	@echo "$(GREEN)✅ Docker Compose环境已停止！$(RESET)"

docker-push: ## 推送Docker镜像到仓库
	@echo "$(GREEN)🐳 推送Docker镜像...$(RESET)"
	docker tag $(DOCKER_IMAGE) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	@echo "$(GREEN)✅ Docker镜像推送完成！$(RESET)"

# 示例运行命令
run-training: ## 运行训练示例
	@echo "$(GREEN)🤖 运行训练示例...$(RESET)"
	cd examples/training/llama_pretraining && $(PYTHON) train_llama.py

run-finetuning: ## 运行微调示例
	@echo "$(GREEN)🎯 运行微调示例...$(RESET)"
	cd examples/finetuning/lora_finetuning && $(PYTHON) lora_finetune.py

run-inference: ## 运行推理示例
	@echo "$(GREEN)⚡ 运行推理示例...$(RESET)"
	cd examples/inference/vllm_serving && $(PYTHON) vllm_server.py

run-hpc: ## 运行HPC计算示例
	@echo "$(GREEN)🔬 运行HPC计算示例...$(RESET)"
	cd examples/hpc/matrix_computation && $(PYTHON) large_matrix_ops.py

# 部署相关命令
deploy-dev: ## 部署到开发环境
	@echo "$(GREEN)🚀 部署到开发环境...$(RESET)"
	bash scripts/deployment/deploy_dev.sh
	@echo "$(GREEN)✅ 开发环境部署完成！$(RESET)"

deploy-prod: ## 部署到生产环境
	@echo "$(GREEN)🚀 部署到生产环境...$(RESET)"
	bash scripts/deployment/deploy_prod.sh
	@echo "$(GREEN)✅ 生产环境部署完成！$(RESET)"

# 监控相关命令
monitor: ## 启动系统监控
	@echo "$(GREEN)📊 启动系统监控...$(RESET)"
	$(PYTHON) common/utils/monitor.py --daemon
	@echo "$(GREEN)✅ 监控服务已启动！$(RESET)"

monitor-stop: ## 停止系统监控
	@echo "$(GREEN)📊 停止系统监控...$(RESET)"
	pkill -f "monitor.py"
	@echo "$(GREEN)✅ 监控服务已停止！$(RESET)"

# 数据相关命令
download-data: ## 下载示例数据集
	@echo "$(GREEN)📥 下载示例数据集...$(RESET)"
	bash scripts/data/download_datasets.sh
	@echo "$(GREEN)✅ 数据集下载完成！$(RESET)"

prepare-data: ## 准备训练数据
	@echo "$(GREEN)📊 准备训练数据...$(RESET)"
	$(PYTHON) scripts/data/prepare_training_data.py
	@echo "$(GREEN)✅ 训练数据准备完成！$(RESET)"

# 模型相关命令
download-models: ## 下载预训练模型
	@echo "$(GREEN)📥 下载预训练模型...$(RESET)"
	bash scripts/models/download_models.sh
	@echo "$(GREEN)✅ 模型下载完成！$(RESET)"

convert-models: ## 转换模型格式
	@echo "$(GREEN)🔄 转换模型格式...$(RESET)"
	$(PYTHON) scripts/models/convert_models.py
	@echo "$(GREEN)✅ 模型转换完成！$(RESET)"

# 配置相关命令
init-config: ## 初始化配置文件
	@echo "$(GREEN)⚙️ 初始化配置文件...$(RESET)"
	cp configs/templates/*.yaml configs/
	@echo "$(GREEN)✅ 配置文件初始化完成！$(RESET)"

validate-config: ## 验证配置文件
	@echo "$(GREEN)✅ 验证配置文件...$(RESET)"
	$(PYTHON) scripts/config/validate_configs.py
	@echo "$(GREEN)✅ 配置文件验证完成！$(RESET)"

# 安全相关命令
security-scan: ## 安全扫描
	@echo "$(GREEN)🔒 运行安全扫描...$(RESET)"
	bandit -r common/ examples/
	safety check
	@echo "$(GREEN)✅ 安全扫描完成！$(RESET)"

# 性能分析命令
profile: ## 性能分析
	@echo "$(GREEN)📈 运行性能分析...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats examples/basic/hello_dcu.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)✅ 性能分析完成！$(RESET)"

# 版本管理命令
version: ## 显示版本信息
	@echo "$(GREEN)📋 版本信息:$(RESET)"
	@echo "项目: $(PROJECT_NAME)"
	@$(PYTHON) -c "import common; print(f'版本: {common.__version__}')" 2>/dev/null || echo "版本: 开发版"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "PyTorch: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"

release: ## 创建发布版本
	@echo "$(GREEN)🏷️ 创建发布版本...$(RESET)"
	bash scripts/release/create_release.sh
	@echo "$(GREEN)✅ 发布版本创建完成！$(RESET)"

# 备份和恢复命令
backup: ## 备份重要数据
	@echo "$(GREEN)💾 备份重要数据...$(RESET)"
	bash scripts/backup/backup_data.sh
	@echo "$(GREEN)✅ 数据备份完成！$(RESET)"

restore: ## 恢复数据
	@echo "$(GREEN)🔄 恢复数据...$(RESET)"
	bash scripts/backup/restore_data.sh
	@echo "$(GREEN)✅ 数据恢复完成！$(RESET)"

# 全面检查命令
check-all: lint test benchmark ## 运行所有检查
	@echo "$(GREEN)✅ 所有检查完成！$(RESET)"

# 快速开始命令
quickstart: install test ## 快速开始（安装+测试）
	@echo "$(GREEN)🎉 快速开始完成！$(RESET)"
	@echo "$(BLUE)接下来您可以:$(RESET)"
	@echo "  make run-training    # 运行训练示例"
	@echo "  make run-finetuning  # 运行微调示例"
	@echo "  make run-inference   # 运行推理示例"
	@echo "  make docker-run      # 启动Docker环境" 