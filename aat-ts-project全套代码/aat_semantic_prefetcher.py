#!/usr/bin/env python3
# aat_semantic_prefetcher.py
"""
AAT语义预取器 - 完整真实数据版本
基于真实BERT模型结构的语义预取，修复统计逻辑
"""

import logging
import numpy as np
from collections import defaultdict, deque
import json
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AAT-Prefetcher")


class SemanticPrefetcher:
    def __init__(self, storage_manager, history_size=100):
        self.storage_manager = storage_manager
        self.history_size = history_size

        # 访问历史记录
        self.access_history = deque(maxlen=history_size)
        self.pattern_counts = defaultdict(int)

        # 修复预取命中统计
        self.prefetch_stats = {
            'prefetched_files': set(),
            'hits': 0,
            'misses': 0,
            'total_prefetches': 0,
            'successful_prefetches': 0
        }

        # 完整的BERT模型层间依赖关系
        self.layer_dependencies = {
            'embedding': ['encoder_layer_0'],
            'encoder_layer_0': ['encoder_layer_1'],
            'encoder_layer_1': ['encoder_layer_2'],
            'encoder_layer_2': ['encoder_layer_3'],
            'encoder_layer_3': ['pooler', 'classifier', 'lm_head'],  # 多个可能的下一层
            'pooler': ['classifier'],
            'classifier': [],
            'lm_head': [],
            'config': ['embedding']  # 配置通常先于嵌入层访问
        }

        # 完整的文件到层映射
        self.file_to_layer = {
            'embedding.bin': 'embedding',
            'layer0.bin': 'encoder_layer_0',
            'layer1.bin': 'encoder_layer_1',
            'layer2.bin': 'encoder_layer_2',
            'layer3.bin': 'encoder_layer_3',
            'output.bin': 'lm_head',
            'pooler.bin': 'pooler',
            'classifier.bin': 'classifier',
            'config.json': 'config',
            # 检查点文件映射
            'checkpoint.ckpt': 'encoder_layer_0',
            'checkpoint_v1.ckpt': 'encoder_layer_1',
            'checkpoint_v2.ckpt': 'encoder_layer_2',
            'checkpoint_v3.ckpt': 'encoder_layer_3',
            'checkpoint_latest.ckpt': 'lm_head'
        }

        # 预取线程池
        self.prefetch_threads = []
        self.running = True

        logger.info("语义预取器初始化完成 - 完整真实数据版本")

    def record_access(self, filename, operation="read", size=0):
        """记录文件访问模式 - 修复预取命中检测"""
        # 首先检查是否是预取命中（在记录访问之前）
        is_prefetch_hit = filename in self.prefetch_stats['prefetched_files']
        if is_prefetch_hit:
            self.record_prefetch_hit(filename)

        access_record = {
            'filename': filename,
            'operation': operation,
            'size': size,
            'timestamp': time.time(),
            'layer_type': self._classify_layer(filename),
            'prefetch_hit': is_prefetch_hit
        }

        self.access_history.append(access_record)

        # 更新模式统计
        if len(self.access_history) > 1:
            prev_record = self.access_history[-2]
            pattern = f"{prev_record['filename']}->{filename}"
            self.pattern_counts[pattern] += 1

        logger.debug(f"记录访问: {filename} -> {access_record['layer_type']} (预取命中: {is_prefetch_hit})")

    def record_prefetch_hit(self, filename):
        """记录预取命中 - 修复统计逻辑"""
        if filename in self.prefetch_stats['prefetched_files']:
            self.prefetch_stats['hits'] += 1
            self.prefetch_stats['prefetched_files'].remove(filename)
            self.prefetch_stats['successful_prefetches'] += 1
            logger.info(f"🎯 预取命中: {filename}")
            return True
        return False

    def _classify_layer(self, filename):
        """根据文件名分类layer类型 - 完整版本"""
        # 优先使用文件映射
        if filename in self.file_to_layer:
            return self.file_to_layer[filename]

        filename_lower = filename.lower()

        if 'embedding' in filename_lower:
            return 'embedding'
        elif 'output' in filename_lower or 'head' in filename_lower:
            return 'lm_head'
        elif 'pooler' in filename_lower:
            return 'pooler'
        elif 'classifier' in filename_lower:
            return 'classifier'
        elif 'config' in filename_lower or 'json' in filename_lower:
            return 'config'
        elif 'layer0' in filename_lower:
            return 'encoder_layer_0'
        elif 'layer1' in filename_lower:
            return 'encoder_layer_1'
        elif 'layer2' in filename_lower:
            return 'encoder_layer_2'
        elif 'layer3' in filename_lower:
            return 'encoder_layer_3'
        elif 'checkpoint' in filename_lower:
            # 检查点文件映射到对应的编码器层
            if 'v1' in filename_lower:
                return 'encoder_layer_1'
            elif 'v2' in filename_lower:
                return 'encoder_layer_2'
            elif 'v3' in filename_lower:
                return 'encoder_layer_3'
            elif 'latest' in filename_lower:
                return 'lm_head'
            else:
                return 'encoder_layer_0'
        else:
            return 'other'

    def predict_next_layers(self, current_file):
        """预测下一个可能访问的layer - 基于完整BERT结构"""
        current_layer = self._classify_layer(current_file)

        # 方法1: 基于预定义的依赖关系
        dependency_based = self.layer_dependencies.get(current_layer, [])

        # 将层名映射回文件名
        dependency_files = []
        for layer in dependency_based:
            # 查找对应的文件名
            for file, file_layer in self.file_to_layer.items():
                if file_layer == layer:
                    dependency_files.append(file)
                    break

        # 方法2: 基于历史访问模式
        pattern_based = self._get_pattern_based_prediction(current_file)

        # 方法3: 基于当前场景的智能预测
        context_based = self._get_context_based_prediction(current_file, current_layer)

        # 合并结果，去重
        predicted_files = list(set(dependency_files + pattern_based + context_based))

        logger.info(f"预测 {current_file}({current_layer}) -> {predicted_files}")
        return predicted_files

    def _get_pattern_based_prediction(self, current_file):
        """基于历史模式预测"""
        predictions = []

        # 查找以当前文件开头的模式
        for pattern, count in self.pattern_counts.items():
            if pattern.startswith(current_file + "->") and count > 1:  # 至少出现2次
                next_file = pattern.split("->")[1]
                predictions.append(next_file)

        return predictions

    def _get_context_based_prediction(self, current_file, current_layer):
        """基于当前场景的智能预测"""
        context_predictions = []

        # 基于BERT推理流程的智能预测
        if current_layer == 'embedding':
            # 嵌入层后通常访问第一个编码器层
            context_predictions.extend(['layer0.bin', 'layer1.bin'])
        elif current_layer.startswith('encoder_layer_'):
            # 编码器层：预测下一个编码器层或输出层
            layer_num = int(current_layer.split('_')[-1])
            if layer_num < 3:  # 假设有4个编码器层 (0-3)
                next_layer = f"layer{layer_num + 1}.bin"
                context_predictions.append(next_layer)
            else:
                # 最后一个编码器层后预测输出层
                context_predictions.extend(['output.bin', 'pooler.bin'])
        elif current_layer == 'pooler':
            # Pooler后通常访问分类器
            context_predictions.append('classifier.bin')

        return context_predictions

    def prefetch_async(self, current_file):
        """异步预取相关文件"""
        if not self.running:
            return

        predicted_files = self.predict_next_layers(current_file)

        for filename in predicted_files:
            # 检查是否已经在预取中
            if filename in self.prefetch_stats['prefetched_files']:
                continue

            thread = threading.Thread(
                target=self._prefetch_file,
                args=(filename,)
            )
            thread.daemon = True
            thread.start()
            self.prefetch_threads.append(thread)

    def _prefetch_file(self, filename):
        """预取单个文件"""
        try:
            # 标记为已预取
            self.prefetch_stats['prefetched_files'].add(filename)
            self.prefetch_stats['total_prefetches'] += 1

            # 检查是否已经在热层
            cached_data = self.storage_manager.get_from_hot_layer(filename)
            if cached_data:
                logger.debug(f"文件已在热层，跳过预取: {filename}")
                self.prefetch_stats['successful_prefetches'] += 1
                return

            # 从真实模型获取数据
            real_data = self.storage_manager.get_real_model_data(filename)
            if real_data:
                # 缓存到热层
                self.storage_manager.cache_to_hot_layer(filename, real_data)
                logger.info(f"✅ 语义预取完成: {filename} ({len(real_data)} bytes)")
                self.prefetch_stats['successful_prefetches'] += 1
            else:
                # 从冷层加载
                cold_data = self.storage_manager.get_from_cold_layer(filename)
                if cold_data:
                    # 缓存到热层
                    self.storage_manager.cache_to_hot_layer(filename, cold_data)
                    logger.info(f"✅ 语义预取完成(冷层): {filename} ({len(cold_data)} bytes)")
                    self.prefetch_stats['successful_prefetches'] += 1
                else:
                    logger.warning(f"预取失败，无数据: {filename}")
                    self.prefetch_stats['misses'] += 1
                    # 从预取集合中移除失败的文件
                    if filename in self.prefetch_stats['prefetched_files']:
                        self.prefetch_stats['prefetched_files'].remove(filename)

        except Exception as e:
            logger.error(f"预取过程出错 {filename}: {e}")
            self.prefetch_stats['misses'] += 1
            # 从预取集合中移除失败的文件
            if filename in self.prefetch_stats['prefetched_files']:
                self.prefetch_stats['prefetched_files'].remove(filename)

    def get_access_patterns(self):
        """获取访问模式统计"""
        return dict(self.pattern_counts)

    def get_prefetch_stats(self):
        """获取预取统计 - 完整修复版本"""
        total_prefetches = self.prefetch_stats['total_prefetches']
        hits = self.prefetch_stats['hits']
        misses = self.prefetch_stats['misses']
        successful = self.prefetch_stats['successful_prefetches']

        total_attempts = hits + misses
        hit_rate = hits / total_attempts if total_attempts > 0 else 0
        success_rate = successful / total_prefetches if total_prefetches > 0 else 0

        return {
            'prefetch_hits': hits,
            'prefetch_misses': misses,
            'prefetch_hit_rate': hit_rate,
            'total_prefetches': total_prefetches,
            'successful_prefetches': successful,
            'prefetch_success_rate': success_rate,
            'pending_prefetches': len(self.prefetch_stats['prefetched_files']),
            'total_attempts': total_attempts
        }

    def save_patterns(self, filepath):
        """保存学习到的模式"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'pattern_counts': dict(self.pattern_counts),
                    'access_history': list(self.access_history),
                    'file_to_layer': self.file_to_layer,
                    'prefetch_stats': self.prefetch_stats,
                    'layer_dependencies': self.layer_dependencies
                }, f, indent=2)
            logger.info(f"模式已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存模式失败: {e}")

    def load_patterns(self, filepath):
        """加载已学习的模式"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.pattern_counts.update(data.get('pattern_counts', {}))
                self.access_history.extend(data.get('access_history', []))
                self.file_to_layer.update(data.get('file_to_layer', {}))
                self.prefetch_stats.update(data.get('prefetch_stats', {
                    'prefetched_files': set(),
                    'hits': 0,
                    'misses': 0,
                    'total_prefetches': 0,
                    'successful_prefetches': 0
                }))
                self.layer_dependencies.update(data.get('layer_dependencies', {}))
            logger.info(f"模式已加载: {filepath}")
        except Exception as e:
            logger.warning(f"加载模式失败: {e}")

    def reset_stats(self):
        """重置统计"""
        self.prefetch_stats = {
            'prefetched_files': set(),
            'hits': 0,
            'misses': 0,
            'total_prefetches': 0,
            'successful_prefetches': 0
        }
        self.pattern_counts.clear()
        self.access_history.clear()
        logger.info("预取统计和模式已重置")

    def stop(self):
        """停止预取器"""
        self.running = False
        for thread in self.prefetch_threads:
            thread.join(timeout=1.0)
        logger.info("语义预取器已停止")


if __name__ == "__main__":
    # 测试预取器
    class MockStorageManager:
        def get_from_hot_layer(self, filename):
            return None

        def get_real_model_data(self, filename):
            return f"real_data_{filename}".encode()

        def cache_to_hot_layer(self, filename, data):
            return True


    storage_manager = MockStorageManager()
    prefetcher = SemanticPrefetcher(storage_manager)

    # 测试预测
    test_files = ['embedding.bin', 'layer0.bin', 'layer1.bin']
    for file in test_files:
        predictions = prefetcher.predict_next_layers(file)
        print(f"{file} -> {predictions}")

    # 测试统计
    stats = prefetcher.get_prefetch_stats()
    print(f"预取统计: {stats}")