#!/usr/bin/env python3
import logging
import gzip
import zlib
import pickle
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AAT-Compression")


class CompressionAlgorithm(Enum):
    GZIP = "gzip"
    ZLIB = "zlib"
    NONE = "none"


class CompressionManager:
    def __init__(self, default_algorithm=CompressionAlgorithm.GZIP):
        self.default_algorithm = default_algorithm
        logger.info(f"压缩管理器初始化，默认算法: {default_algorithm.value}")

    def compress(self, data, algorithm=None):
        """压缩数据"""
        if algorithm is None:
            algorithm = self.default_algorithm

        if algorithm == CompressionAlgorithm.NONE or not data:
            return data, algorithm

        try:
            if algorithm == CompressionAlgorithm.GZIP:
                compressed = gzip.compress(data)
            elif algorithm == CompressionAlgorithm.ZLIB:
                compressed = zlib.compress(data)
            else:
                raise ValueError(f"不支持的压缩算法: {algorithm}")

            # 计算压缩率
            original_size = len(data)
            compressed_size = len(compressed)
            compression_ratio = compressed_size / original_size if original_size > 0 else 0

            logger.debug(f"压缩完成: {original_size} -> {compressed_size} "
                         f"(比率: {compression_ratio:.2f})")

            return compressed, algorithm

        except Exception as e:
            logger.error(f"压缩失败: {e}")
            return data, CompressionAlgorithm.NONE

    def decompress(self, data, algorithm):
        """解压数据"""
        if algorithm == CompressionAlgorithm.NONE or not data:
            return data

        try:
            if algorithm == CompressionAlgorithm.GZIP:
                decompressed = gzip.decompress(data)
            elif algorithm == CompressionAlgorithm.ZLIB:
                decompressed = zlib.decompress(data)
            else:
                raise ValueError(f"不支持的压缩算法: {algorithm}")

            logger.debug(f"解压完成: {len(data)} -> {len(decompressed)}")
            return decompressed

        except Exception as e:
            logger.error(f"解压失败: {e}")
            return data

    def should_compress(self, data, min_savings=0.1):
        """判断是否值得压缩"""
        if not data or len(data) < 1024:  # 小于1KB不压缩
            return False

        # 测试压缩
        test_compressed, _ = self.compress(data[:min(8192, len(data))])
        savings = 1 - (len(test_compressed) / len(data))

        return savings >= min_savings