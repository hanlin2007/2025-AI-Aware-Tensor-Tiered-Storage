#!/usr/bin/env python3
import os
import sys
import logging
import time
import stat
import errno

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import fuse
from aat_storage_manager_v2 import AATStorageManagerV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AAT-FUSE-V2")

FUSE = fuse.FUSE
Operations = fuse.Operations
FuseOSError = fuse.FuseOSError


class AATFUSEV2(Operations):
    def __init__(self):
        self.storage_manager = AATStorageManagerV2()
        logger.info("AAT智能FUSE文件系统V2初始化完成")

        # 虚拟文件系统结构
        self.files = self._create_virtual_filesystem()
        self.file_data_cache = {}

    def _create_virtual_filesystem(self):
        """创建虚拟文件系统"""
        base_attrs = {
            'st_mode': 0,
            'st_size': 0,
            'st_ctime': time.time(),
            'st_mtime': time.time(),
            'st_atime': time.time(),
            'st_nlink': 1,
            'st_uid': os.getuid(),
            'st_gid': os.getgid(),
        }

        files = {
            '/': {**base_attrs, 'st_mode': stat.S_IFDIR | 0o755, 'st_size': 4096},

            # 模型文件
            '/embedding.bin': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 512 * 1024},
            '/layer0.bin': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 2 * 1024 * 1024},
            '/layer1.bin': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 4 * 1024 * 1024},
            '/layer2.bin': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 8 * 1024 * 1024},
            '/layer3.bin': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 16 * 1024 * 1024},
            '/output.bin': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 1 * 1024 * 1024},

            # 配置和检查点
            '/config.json': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 1024},
            '/checkpoint.ckpt': {**base_attrs, 'st_mode': stat.S_IFREG | 0o644, 'st_size': 50 * 1024 * 1024},
        }

        return files

    def getattr(self, path, fh=None):
        logger.info(f"getattr: {path}")
        if path in self.files:
            attrs = self.files[path].copy()
            attrs.setdefault('st_blksize', 4096)
            attrs.setdefault('st_blocks', (attrs['st_size'] + 511) // 512)
            return attrs
        raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        logger.info(f"readdir: {path}")
        if path == '/':
            return ['.', '..'] + [f[1:] for f in self.files.keys() if f != '/']
        raise FuseOSError(errno.ENOENT)

    def open(self, path, flags):
        logger.info(f"open: {path} (flags: {flags})")
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        if flags & 3 != 0:  # 写操作
            raise FuseOSError(errno.EROFS)
        return 0

    def read(self, path, size, offset, fh):
        logger.info(f"read: {path}, size: {size}, offset: {offset}")

        if path not in self.files:
            raise FuseOSError(errno.ENOENT)

        filename = os.path.basename(path)

        try:
            # 使用智能存储管理器获取数据
            data = self.storage_manager.get_data(filename, size, offset)
            logger.info(f"✓ 返回数据: {len(data)} bytes")
            return data

        except Exception as e:
            logger.error(f"读取失败 {path}: {e}")
            return b''

    def statfs(self, path):
        return {
            'f_bsize': 4096,
            'f_blocks': 1000000,
            'f_bavail': 999000,
            'f_files': 1000000,
            'f_ffree': 999000,
            'f_namemax': 255
        }


def main():
    if len(sys.argv) != 2:
        print("用法: python aat_fuse_v2.py <挂载点>")
        sys.exit(1)

    mount_point = sys.argv[1]

    print("=" * 50)
    print("启动AAT智能FUSE文件系统V2")
    print(f"挂载点: {mount_point}")
    print("=" * 50)

    try:
        FUSE(AATFUSEV2(), mount_point, foreground=True, nothreads=True)
    except Exception as e:
        logger.error(f"FUSE挂载失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()