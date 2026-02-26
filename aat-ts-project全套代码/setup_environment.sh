#!/bin/bash
echo "=== 智能环境设置脚本 ==="

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ 请先激活虚拟环境: source venv/bin/activate"
    exit 1
fi

echo "虚拟环境: $VIRTUAL_ENV"

# 函数：检查包是否已安装
check_and_install_pip() {
    local package_name=$1
    local pip_package_name=$2
    local install_args=$3

    # 如果没有指定pip包名，使用与检查相同的包名
    if [ -z "$pip_package_name" ]; then
        pip_package_name="$package_name"
    fi

    if pip show "$package_name" &> /dev/null; then
        echo "✅ $package_name 已安装，跳过"
    else
        echo "📦 安装 $package_name..."
        if [ -z "$install_args" ]; then
            pip install "$pip_package_name"
        else
            pip install "$pip_package_name" $install_args
        fi
    fi
}

# 函数：检查系统服务是否运行
check_and_start_service() {
    local service_name=$1

    if systemctl is-active --quiet "$service_name"; then
        echo "✅ $service_name 服务正在运行"
    else
        echo "🔧 启动 $service_name 服务..."
        sudo systemctl start "$service_name"
        sudo systemctl enable "$service_name"
    fi
}

echo "1. 检查并安装系统依赖..."
if ! command -v redis-server &> /dev/null; then
    echo "📦 安装Redis服务器..."
    sudo apt update
    sudo apt install redis-server -y
else
    echo "✅ Redis服务器 已安装"
fi

echo "2. 检查并配置Redis服务..."
check_and_start_service "redis"

echo "3. 检查并安装Python依赖..."
# 注意：有些包在pip中的名称与import名称不同
check_and_install_pip "torch" "torch" "--index-url https://download.pytorch.org/whl/cpu"
check_and_install_pip "transformers" "transformers"
check_and_install_pip "minio" "minio"
check_and_install_pip "redis" "redis"
check_and_install_pip "sklearn" "scikit-learn"  # pip包名是scikit-learn，但import是sklearn
check_and_install_pip "fuse" "fuse-python"  # pip包名是fuse-python，但import是fuse

# 安装numpy，因为torch需要它
check_and_install_pip "numpy" "numpy"

echo "4. 验证环境..."
python check_environment.py

echo "=== 环境设置完成 ==="