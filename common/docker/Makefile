# 海光DCU加速卡实战项目 Makefile
# 提供便捷的开发和部署命令

.PHONY: help build run stop clean test docs setup check install jupyter inference train monitor

# 默认目标
.DEFAULT_GOAL := help

# 项目配置
PROJECT_NAME := dcu-in-action
IMAGE_NAME := $(PROJECT_NAME):latest
CONTAINER_NAME := dcu-dev-main

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# 帮助信息
help: ## 显示帮助信息
	@echo "$(BLUE)海光DCU加速卡实战项目 - 管理命令$(NC)"
	@echo ""
	@echo "$(YELLOW)Docker 相关命令:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(build|run|stop|clean)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)开发 相关命令:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(setup|check|install|test)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)应用 相关命令:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(jupyter|inference|train|monitor)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

# Docker相关命令
build: ## 构建Docker镜像
	@echo "$(BLUE)构建Docker镜像...$(NC)"
	docker build -t $(IMAGE_NAME) .
	@echo "$(GREEN)镜像构建完成: $(IMAGE_NAME)$(NC)"

run: ## 启动主开发容器
	@echo "$(BLUE)启动DCU开发环境...$(NC)"
	docker-compose up -d dcu-dev
	@echo "$(GREEN)容器已启动，可通过以下方式访问:$(NC)"
	@echo "  终端: docker exec -it $(CONTAINER_NAME) bash"
	@echo "  Jupyter: http://localhost:8888"

run-all: ## 启动所有服务
	@echo "$(BLUE)启动所有服务...$(NC)"
	docker-compose --profile jupyter --profile inference --profile monitor up -d
	@echo "$(GREEN)所有服务已启动$(NC)"

stop: ## 停止所有容器
	@echo "$(BLUE)停止所有容器...$(NC)"
	docker-compose down
	@echo "$(GREEN)容器已停止$(NC)"

restart: stop run ## 重启容器

clean: ## 清理Docker资源
	@echo "$(BLUE)清理Docker资源...$(NC)"
	docker-compose down -v --remove-orphans
	docker image prune -f
	@echo "$(GREEN)清理完成$(NC)"

clean-all: ## 深度清理（包括镜像）
	@echo "$(RED)深度清理Docker资源...$(NC)"
	docker-compose down -v --remove-orphans
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
	docker system prune -af
	@echo "$(GREEN)深度清理完成$(NC)"

# 开发相关命令
setup: ## 初始化开发环境
	@echo "$(BLUE)初始化开发环境...$(NC)"
	@mkdir -p models datasets outputs logs
	@chmod +x scripts/setup/*.sh scripts/utils/*.py examples/*/*.py
	@chmod +x docker-entrypoint.sh
	@echo "$(GREEN)开发环境初始化完成$(NC)"

check: ## 检查DCU环境
	@echo "$(BLUE)检查DCU环境...$(NC)"
	@if [ -f scripts/setup/check_environment.sh ]; then \
		bash scripts/setup/check_environment.sh; \
	else \
		echo "$(RED)环境检查脚本不存在$(NC)"; \
	fi

install: ## 安装项目依赖
	@echo "$(BLUE)安装项目依赖...$(NC)"
	@if [ -f scripts/setup/install_dependencies.sh ]; then \
		bash scripts/setup/install_dependencies.sh; \
	else \
		pip install -r requirements.txt; \
	fi
	@echo "$(GREEN)依赖安装完成$(NC)"

test: ## 运行测试套件
	@echo "$(BLUE)运行测试套件...$(NC)"
	@if command -v pytest >/dev/null 2>&1; then \
		pytest tests/ -v; \
	else \
		python -m pytest tests/ -v; \
	fi

test-dcu: ## 测试DCU环境
	@echo "$(BLUE)测试DCU环境...$(NC)"
	python examples/llm-inference/simple_test.py

lint: ## 代码格式检查
	@echo "$(BLUE)运行代码格式检查...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black --check .; \
	fi
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 .; \
	fi

format: ## 格式化代码
	@echo "$(BLUE)格式化代码...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black .; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort .; \
	fi

# 应用相关命令
jupyter: ## 启动Jupyter Lab
	@echo "$(BLUE)启动Jupyter Lab...$(NC)"
	docker-compose --profile jupyter up -d dcu-jupyter
	@echo "$(GREEN)Jupyter Lab已启动: http://localhost:8889$(NC)"

jupyter-local: ## 本地启动Jupyter Lab
	@echo "$(BLUE)本地启动Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

inference: ## 启动推理服务
	@echo "$(BLUE)启动推理服务...$(NC)"
	docker-compose --profile inference up -d dcu-inference
	@echo "$(GREEN)推理服务已启动: http://localhost:8001$(NC)"

inference-local: ## 本地启动推理服务
	@echo "$(BLUE)本地启动推理服务...$(NC)"
	python examples/llm-inference/vllm_server.py --mode server

train: ## 启动训练容器
	@echo "$(BLUE)启动训练容器...$(NC)"
	docker-compose --profile training run --rm dcu-training bash

train-llama: ## 训练LLaMA模型
	@echo "$(BLUE)启动LLaMA训练...$(NC)"
	python examples/llm-training/train_llama.py \
		--output_dir ./outputs/llama-trained \
		--num_train_epochs 1 \
		--per_device_train_batch_size 2

train-lora: ## LoRA微调
	@echo "$(BLUE)启动LoRA微调...$(NC)"
	python examples/llm-fine-tuning/lora_finetune.py \
		--create_sample_data
	python examples/llm-fine-tuning/lora_finetune.py \
		--dataset_path ./data/sample_data.json \
		--output_dir ./outputs/lora-finetuned

monitor: ## 启动监控服务
	@echo "$(BLUE)启动DCU监控...$(NC)"
	docker-compose --profile monitor up -d dcu-monitor

monitor-local: ## 本地启动监控
	@echo "$(BLUE)本地启动DCU监控...$(NC)"
	python scripts/utils/monitor_performance.py monitor

# 进入容器
shell: ## 进入开发容器
	@echo "$(BLUE)进入开发容器...$(NC)"
	docker exec -it $(CONTAINER_NAME) bash

shell-root: ## 以root身份进入容器
	@echo "$(BLUE)以root身份进入容器...$(NC)"
	docker exec -it --user root $(CONTAINER_NAME) bash

# 日志查看
logs: ## 查看容器日志
	@echo "$(BLUE)查看容器日志...$(NC)"
	docker-compose logs -f dcu-dev

logs-all: ## 查看所有服务日志
	@echo "$(BLUE)查看所有服务日志...$(NC)"
	docker-compose logs -f

# 备份和恢复
backup: ## 备份重要数据
	@echo "$(BLUE)备份重要数据...$(NC)"
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf backup_$$timestamp.tar.gz \
		models/ datasets/ outputs/ docs/ examples/ scripts/ \
		requirements.txt pyproject.toml README.md \
		--exclude='*.pyc' --exclude='__pycache__' --exclude='*.log'
	@echo "$(GREEN)备份完成: backup_$$(date +%Y%m%d_%H%M%S).tar.gz$(NC)"

# 文档生成
docs: ## 生成项目文档
	@echo "$(BLUE)生成项目文档...$(NC)"
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs/ docs/_build/; \
	else \
		echo "$(YELLOW)Sphinx未安装，跳过文档生成$(NC)"; \
	fi

# 性能测试
benchmark: ## 运行性能基准测试
	@echo "$(BLUE)运行性能基准测试...$(NC)"
	python scripts/utils/monitor_performance.py benchmark

benchmark-inference: ## 推理性能测试
	@echo "$(BLUE)推理性能测试...$(NC)"
	python examples/llm-inference/chatglm_inference.py --mode benchmark

# 部署相关
deploy-dev: build run ## 开发环境部署
	@echo "$(GREEN)开发环境部署完成$(NC)"

deploy-prod: ## 生产环境部署
	@echo "$(BLUE)部署生产环境...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "$(GREEN)生产环境部署完成$(NC)"

# 状态检查
status: ## 查看服务状态
	@echo "$(BLUE)服务状态:$(NC)"
	@docker-compose ps

health: ## 健康检查
	@echo "$(BLUE)执行健康检查...$(NC)"
	@docker exec $(CONTAINER_NAME) python -c "import torch; print('DCU可用:', torch.cuda.is_available())" 2>/dev/null || echo "$(RED)容器未运行$(NC)"

# 版本信息
version: ## 显示版本信息
	@echo "$(BLUE)版本信息:$(NC)"
	@echo "项目版本: $$(grep version pyproject.toml | head -1 | cut -d'"' -f2)"
	@echo "Docker镜像: $(IMAGE_NAME)"
	@echo "构建时间: $$(docker inspect $(IMAGE_NAME) --format='{{.Created}}' 2>/dev/null || echo '未构建')"

# 清理生成的文件
clean-files: ## 清理生成的文件
	@echo "$(BLUE)清理生成的文件...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .coverage
	@echo "$(GREEN)文件清理完成$(NC)"

# 环境信息
info: ## 显示环境信息
	@echo "$(BLUE)环境信息:$(NC)"
	@echo "操作系统: $$(uname -s)"
	@echo "架构: $$(uname -m)"
	@echo "Docker版本: $$(docker --version 2>/dev/null || echo '未安装')"
	@echo "Docker Compose版本: $$(docker-compose --version 2>/dev/null || echo '未安装')"
	@echo "Python版本: $$(python --version 2>/dev/null || echo '未安装')"
	@echo "项目目录: $$(pwd)" 