#!/bin/bash
# AAT-TS 演示环境停止脚本

echo "🛑 停止 AAT-TS 演示环境..."
echo "=========================================="

# 停止 FUSE
if [ -f "fuse.pid" ]; then
    FUSE_PID=$(cat fuse.pid)
    kill $FUSE_PID 2>/dev/null && echo "✅ 停止 FUSE 文件系统"
    rm -f fuse.pid
fi

# 停止 MinIO
if [ -f "minio.pid" ]; then
    MINIO_PID=$(cat minio.pid)
    kill $MINIO_PID 2>/dev/null && echo "✅ 停止 MinIO"
    rm -f minio.pid
fi

# 清理挂载点
sudo umount /mnt/aat 2>/dev/null && echo "✅ 卸载挂载点"
sudo fusermount -u /mnt/aat 2>/dev/null || true

# 清理日志文件
rm -f minio.log fuse.log 2>/dev/null && echo "✅ 清理日志文件"

echo ""
echo "✅ 所有服务已停止"
echo "=========================================="