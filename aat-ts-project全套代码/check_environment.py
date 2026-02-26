# check_environment.py
import sys
import subprocess

def check_module(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - 错误: {e}")
        return False

# 模块列表 - 使用实际的import名称
modules = ['torch', 'transformers', 'minio', 'redis', 'fuse', 'sklearn']
print("检查Python模块...")
all_ok = all(check_module(m) for m in modules)

print("\n检查服务...")
# 检查Redis连接
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=3)
    r.ping()
    print("✅ Redis连接正常")
except Exception as e:
    print(f"❌ Redis连接失败: {e}")

print(f"\n环境检查: {'通过' if all_ok else '失败'}")