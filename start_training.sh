#!/bin/bash
# -*- coding: utf-8 -*-
#
# 神机网络安全模型自动化训练启动脚本
#
# 使用方法:
# 1. 完整训练: ./start_training.sh
# 2. 仅数据处理: ./start_training.sh --mode data
# 3. 仅模型训练: ./start_training.sh --mode train
# 4. 仅模型测试: ./start_training.sh --mode test
# 5. 交互模式: ./start_training.sh --mode interactive
#

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查系统环境
check_system() {
    log_step "检查系统环境..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python版本: $python_version"
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 未安装"
        exit 1
    fi
    
    # screen检查已移除，统一在前台运行
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_info "检测到NVIDIA GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    else
        log_warn "未检测到NVIDIA GPU，将使用CPU训练"
    fi
    
    log_info "系统环境检查完成"
}

# 配置pip镜像源
configure_pip_mirror() {
    log_step "配置pip镜像源..."
    
    # 测试网络连接并选择最佳镜像源
    if ping -c 1 -W 3 pypi.tuna.tsinghua.edu.cn &> /dev/null; then
        log_info "配置清华大学镜像源"
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    elif ping -c 1 -W 3 mirrors.aliyun.com &> /dev/null; then
        log_info "配置阿里云镜像源"
        pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
        pip config set global.trusted-host mirrors.aliyun.com
    elif ping -c 1 -W 3 pypi.douban.com &> /dev/null; then
        log_info "配置豆瓣镜像源"
        pip config set global.index-url https://pypi.douban.com/simple/
        pip config set global.trusted-host pypi.douban.com
    else
        log_warn "网络连接检查失败，使用默认源（可能较慢）"
    fi
    
    # 设置其他pip优化参数
    pip config set global.timeout 300
    pip config set global.retries 3
    
    log_info "当前pip配置:"
    pip config list || log_warn "无法显示pip配置"
}

# 设置虚拟环境
setup_venv() {
    log_step "设置Python虚拟环境..."
    
    VENV_DIR="./venv"
    
    if [ ! -d "$VENV_DIR" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
    fi
    
    log_info "激活虚拟环境..."
    source "$VENV_DIR/bin/activate"
    
    # 配置pip镜像源
    configure_pip_mirror
    
    # 升级pip和构建工具
    log_info "升级pip、setuptools和wheel..."
    pip install --upgrade pip setuptools wheel || {
        log_warn "构建工具升级失败，继续使用当前版本"
    }
    
    log_info "虚拟环境设置完成"
}

# 安装依赖
install_dependencies() {
    log_step "安装Python依赖..."
    
    # 优先使用基础依赖文件，避免安装问题
    if [ -f "requirements-basic.txt" ]; then
        log_info "使用requirements-basic.txt安装核心依赖..."
        pip install -r requirements-basic.txt || {
            log_error "基础依赖安装失败"
            exit 1
        }
        
        # 尝试安装可选依赖
        log_info "安装可选依赖..."
        
        # 安装bitsandbytes（量化支持）
        pip install bitsandbytes>=0.39.0 || {
            log_warn "bitsandbytes安装失败，量化功能可能不可用"
        }
        
        # 安装nvidia-ml-py（GPU监控）
        if command -v nvidia-smi &> /dev/null; then
            pip install nvidia-ml-py>=12.535.108 || {
                log_warn "nvidia-ml-py安装失败，GPU监控功能可能不可用"
            }
            
            # 尝试安装flash-attn
            log_info "检测到NVIDIA GPU，尝试安装flash-attn（可选）..."
            log_info "正在预安装torch以支持flash-attn编译..."
            pip install torch>=2.0.0 || log_warn "torch预安装失败，flash-attn可能无法安装"
            pip install flash-attn>=2.0.0 --no-build-isolation || {
                log_warn "flash-attn安装失败，将跳过此依赖（不影响基本功能）"
                log_warn "如需flash-attn，请手动安装：pip install flash-attn --no-build-isolation"
            }
        else
            log_info "未检测到GPU，跳过GPU相关可选依赖"
        fi
        
    elif [ -f "requirements.txt" ]; then
        log_info "使用requirements.txt安装依赖..."
        pip install -r requirements.txt || {
            log_error "依赖安装失败，请检查requirements.txt"
            exit 1
        }
    else
        log_error "未找到依赖文件 (requirements-basic.txt 或 requirements.txt)"
        exit 1
    fi
    
    log_info "依赖安装完成"
}

# 设置环境变量
setup_environment() {
    log_step "设置环境变量..."
    
    # 设置HuggingFace缓存目录
    export HF_HOME="./cache/huggingface"
    export TRANSFORMERS_CACHE="./cache/transformers"
    
    # 设置ModelScope缓存目录
    export MODELSCOPE_CACHE="./cache/modelscope"
    
    # 设置CUDA相关环境变量
    export CUDA_VISIBLE_DEVICES=0
    
    # 设置Python路径
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    
    log_info "环境变量设置完成"
}

# 创建日志目录
setup_logging() {
    log_step "设置日志目录..."
    
    LOG_DIR="./logs"
    mkdir -p "$LOG_DIR"
    
    # 生成日志文件名
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"
    
    log_info "日志文件: $LOG_FILE"
}

# 运行训练
run_training() {
    local mode="$1"
    local additional_args="$2"
    
    log_step "开始模型训练..."
    
    # 构建命令
    CMD="python3 main.py --mode $mode $additional_args"
    
    log_info "执行命令: $CMD"
    
    # 前台运行训练
    log_info "在前台运行训练..."
    log_info "提示: 如需后台运行，可使用 screen 或 nohup 命令"
    $CMD 2>&1 | tee "$LOG_FILE"
}

# 显示帮助信息
show_help() {
    echo "神机网络安全模型自动化训练系统"
    echo ""
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --mode MODE          运行模式 (full|data|train|test|interactive|check)"
    echo "  --force-download     强制重新下载数据"
    echo "  --model-path PATH    模型路径 (用于test和interactive模式)"
    echo "  --resume             从最新checkpoint继续训练"
    echo "  --resume-from PATH   从指定checkpoint路径继续训练"
    echo "  --model MODEL        选择基础模型 (qwen|chatglm|baichuan|llama等)"
    echo "  --list-models        列出支持的模型"
    echo "  --help              显示此帮助信息"
    echo ""
    echo "运行模式:"
    echo "  full                 完整训练流程 (默认)"
    echo "  data                 仅数据下载和处理"
    echo "  train                仅模型训练"
    echo "  test                 仅模型测试"
    echo "  interactive          交互式对话"
    echo "  check                检查系统环境"
    echo ""
    echo "示例:"
    echo "  $0                                    # 完整训练"
    echo "  $0 --mode data --force-download       # 重新下载数据"
    echo "  $0 --mode train                       # 仅训练模型"
    echo "  $0 --mode train --resume              # 从最新checkpoint继续训练"
    echo "  $0 --mode train --model chatglm       # 使用ChatGLM模型训练"
    echo "  $0 --list-models                      # 列出支持的模型"
    echo "  $0 --mode test                        # 测试模型"
    echo "  $0 --mode interactive                 # 交互模式"
}

# 主函数
main() {
    # 默认参数
    MODE="full"
    ADDITIONAL_ARGS=""
    
    # 如果没有提供任何参数，显示帮助信息
    if [[ $# -eq 0 ]]; then
        show_help
        echo ""
        echo "提示: 如果要运行完整训练流程，请使用: $0 --mode full"
        exit 0
    fi
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --force-download)
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --force-download"
                shift
                ;;
            --model-path)
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --model-path '$2'"
                shift 2
                ;;
            --resume)
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --resume"
                shift
                ;;
            --resume-from)
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --resume-from '$2'"
                shift 2
                ;;
            --model)
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --model '$2'"
                shift 2
                ;;
            --list-models)
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --list-models"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证模式
    case $MODE in
        full|data|train|test|interactive|check)
            ;;
        *)
            log_error "无效的运行模式: $MODE"
            show_help
            exit 1
            ;;
    esac
    
    echo "==========================================="
    echo "神机网络安全模型自动化训练系统"
    echo "==========================================="
    echo "运行模式: $MODE"
    echo "开始时间: $(date)"
    echo "==========================================="
    
    # 执行步骤
    check_system
    setup_venv
    install_dependencies
    setup_environment
    setup_logging
    
    # 运行训练
    run_training "$MODE" "$ADDITIONAL_ARGS"
    
    log_info "训练完成"
}

# 错误处理
trap 'log_error "脚本执行失败，退出码: $?"' ERR

# 运行主函数
main "$@"