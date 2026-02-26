#!/bin/bash
# AAT-TS 演示一键启动脚本
# 保存为 start_demo.sh，然后 chmod +x start_demo.sh

echo "🚀 启动 AAT-TS 演示环境..."
echo "=========================================="

# 检查并激活虚拟环境
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ 虚拟环境不存在，请先创建虚拟环境"
    exit 1
fi

source venv/bin/activate

# 函数：清理旧进程
cleanup_old_processes() {
    echo "🧹 清理旧进程..."
    sudo umount /mnt/aat 2>/dev/null || true
    sudo fusermount -u /mnt/aat 2>/dev/null || true
    pkill -f "minio server" 2>/dev/null || true
    pkill -f "python aat_fuse_v2.py" 2>/dev/null || true
    sleep 2
}

# 函数：检查服务状态
check_service() {
    local service_name=$1
    local check_command=$2
    if eval $check_command >/dev/null 2>&1; then
        echo "✅ $service_name 运行正常"
        return 0
    else
        echo "❌ $service_name 未运行"
        return 1
    fi
}

# 函数：等待服务启动
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=10
    local attempt=1

    echo "⏳ 等待 $service_name 启动..."
    while [ $attempt -le $max_attempts ]; do
        if eval $check_command >/dev/null 2>&1; then
            echo "✅ $service_name 已就绪"
            return 0
        fi
        echo "   尝试 $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done
    echo "❌ $service_name 启动超时"
    return 1
}

# 主启动流程
main() {
    echo "📋 启动流程开始..."

    # 1. 清理旧进程
    cleanup_old_processes

    # 2. 启动 Redis (使用系统服务)
    echo "1. 检查 Redis..."
    sudo systemctl start redis >/dev/null 2>&1
    sleep 1
    check_service "Redis" "redis-cli ping"

    # 3. 启动 MinIO (后台运行)
    echo "2. 启动 MinIO..."
    mkdir -p ~/minio-data
    nohup minio server ~/minio-data --console-address ":9001" > minio.log 2>&1 &
    MINIO_PID=$!
    echo $MINIO_PID > minio.pid

    wait_for_service "MinIO" "curl -s http://127.0.0.1:9000/minio/health/live >/dev/null"

    # 4. 准备挂载点
    echo "3. 准备挂载点..."
    sudo mkdir -p /mnt/aat
    sudo chown $USER:$USER /mnt/aat 2>/dev/null || true

    # 5. 启动 FUSE 文件系统 (后台运行)
    echo "4. 启动 FUSE 文件系统..."
    nohup python aat_fuse_v2.py /mnt/aat > fuse.log 2>&1 &
    FUSE_PID=$!
    echo $FUSE_PID > fuse.pid

    wait_for_service "FUSE" "ls /mnt/aat >/dev/null 2>&1"

    # 6. 验证所有服务
    echo "5. 验证服务状态..."
    check_service "Redis" "redis-cli ping"
    check_service "MinIO" "curl -s http://127.0.0.1:9000/minio/health/live >/dev/null"
    check_service "FUSE" "ls /mnt/aat >/dev/null 2>&1"

    # 7. 显示状态信息
    echo ""
    echo "🎉 AAT-TS 演示环境启动完成!"
    echo "=========================================="
    echo "📊 服务状态:"
    echo "   Redis: http://localhost:6379"
    echo "   MinIO: http://localhost:9001 (管理界面)"
    echo "   FUSE:  /mnt/aat (挂载点)"
    echo ""
    echo "🔧 可用命令:"
    echo "   运行测试: python aat_final_test.py"
    echo "   查看文件: ls /mnt/aat"
    echo "   停止服务: ./stop_demo.sh"
    echo "=========================================="

    # 8. 等待用户输入，然后运行测试
    read -p "按回车键运行综合测试，或 Ctrl+C 退出..."
    echo "🧪 运行综合测试..."
    python aat_final_test.py
}

# 执行主函数
main