# check_dependencies.py
# !/usr/bin/env python3
import importlib
import sys


def check_package(package_name, min_version=None):
    try:
        module = importlib.import_module(package_name)
        if min_version:
            version = getattr(module, '__version__', '未知')
            print(f"✅ {package_name}: {version} (需要 {min_version}+)")
        else:
            print(f"✅ {package_name}: 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False


def main():
    print("🔍 检查项目依赖...")
    print("=" * 50)

    dependencies = [
        ("redis", "4.5.0"),
        ("minio", "7.1.0"),
        ("numpy", "1.21.0"),
        ("matplotlib", "3.5.0"),
        ("yaml", "6.0"),  # PyYAML
        ("fuse", "1.0.0")  # fuse-python
    ]

    all_ok = True
    for package, version in dependencies:
        if not check_package(package, version):
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("✅ 所有依赖已安装!")
    else:
        print("❌ 部分依赖缺失，请运行: pip install -r requirements.txt")

    return all_ok


if __name__ == "__main__":
    main()