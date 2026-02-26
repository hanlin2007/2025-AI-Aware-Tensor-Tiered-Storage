# aat_real_workload_generator.py
# !/usr/bin/env python3
import logging
import numpy as np
import time
import random
from collections import deque
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AAT-Workload-Generator")


class RealWorkloadGenerator:
    def __init__(self):
        self.workload_patterns = {
            'online_inference': self._generate_online_inference_pattern,
            'edge_finetuning': self._generate_edge_finetuning_pattern,
            'research_environment': self._generate_research_pattern
        }

        # 真实访问模式统计
        self.access_statistics = {
            'total_requests': 0,
            'layer_access_count': {},
            'pattern_frequency': {}
        }

    def generate_workload(self, scenario, duration_seconds=60, intensity='medium'):
        """生成指定场景的工作负载"""
        if scenario not in self.workload_patterns:
            raise ValueError(f"不支持的场景: {scenario}")

        logger.info(f"生成 {scenario} 工作负载，强度: {intensity}")

        workload = {
            'scenario': scenario,
            'intensity': intensity,
            'requests': [],
            'start_time': time.time(),
            'duration': duration_seconds
        }

        pattern_generator = self.workload_patterns[scenario]
        requests = pattern_generator(duration_seconds, intensity)
        workload['requests'] = requests

        # 更新统计
        self._update_statistics(requests, scenario)

        return workload

    def _generate_online_inference_pattern(self, duration, intensity):
        """生成在线推理工作负载模式"""
        # 强度参数
        intensity_params = {
            'low': {'qps': 5, 'burst_prob': 0.1},
            'medium': {'qps': 20, 'burst_prob': 0.3},
            'high': {'qps': 100, 'burst_prob': 0.6}
        }
        params = intensity_params[intensity]

        requests = []
        current_time = 0
        request_id = 0

        # 推理层访问模式
        inference_patterns = [
            ['embedding', 'layer0', 'output'],  # 简单查询
            ['embedding', 'layer0', 'layer1', 'output'],  # 中等查询
            ['embedding', 'layer0', 'layer1', 'layer2', 'output']  # 复杂查询
        ]

        pattern_weights = [0.6, 0.3, 0.1]  # 模式权重

        while current_time < duration:
            # 生成请求间隔（泊松分布）
            interval = np.random.exponential(1.0 / params['qps'])
            current_time += interval

            # 选择访问模式
            pattern = random.choices(inference_patterns, weights=pattern_weights)[0]

            # 突发流量模拟
            if random.random() < params['burst_prob']:
                burst_requests = random.randint(2, 5)
                for _ in range(burst_requests):
                    burst_pattern = random.choices(inference_patterns, weights=pattern_weights)[0]
                    for layer in burst_pattern:
                        requests.append({
                            'id': request_id,
                            'timestamp': current_time + random.uniform(0, 0.01),
                            'layer': layer,
                            'operation': 'read',
                            'size': self._get_layer_size(layer),
                            'pattern': 'burst'
                        })
                        request_id += 1

            # 正常请求
            for layer in pattern:
                requests.append({
                    'id': request_id,
                    'timestamp': current_time,
                    'layer': layer,
                    'operation': 'read',
                    'size': self._get_layer_size(layer),
                    'pattern': 'normal'
                })
                request_id += 1

        # 按时间排序
        requests.sort(key=lambda x: x['timestamp'])
        return requests

    def _generate_edge_finetuning_pattern(self, duration, intensity):
        """生成边缘微调工作负载模式"""
        requests = []
        current_time = 0
        request_id = 0

        # 微调模式：频繁读取参数，偶尔保存检查点
        while current_time < duration:
            # 训练阶段：频繁读取各层
            training_duration = random.uniform(5, 15)
            training_end = current_time + training_duration

            while current_time < training_end and current_time < duration:
                # 读取模式
                layers = ['embedding', 'layer0', 'layer1', 'output']
                for layer in random.sample(layers, random.randint(2, 4)):
                    requests.append({
                        'id': request_id,
                        'timestamp': current_time,
                        'layer': layer,
                        'operation': 'read',
                        'size': self._get_layer_size(layer),
                        'pattern': 'training_read'
                    })
                    request_id += 1
                    current_time += random.uniform(0.1, 0.5)

            # 保存检查点
            if current_time < duration:
                requests.append({
                    'id': request_id,
                    'timestamp': current_time,
                    'layer': 'checkpoint',
                    'operation': 'write',
                    'size': random.randint(50 * 1024 * 1024, 200 * 1024 * 1024),  # 50-200MB
                    'pattern': 'checkpoint_save'
                })
                request_id += 1
                current_time += random.uniform(1, 3)

        return requests

    def _generate_research_pattern(self, duration, intensity):
        """生成科研环境工作负载模式"""
        requests = []
        current_time = 0
        request_id = 0

        # 模型版本
        versions = [f'checkpoint_v{i}' for i in range(1, 6)] + ['checkpoint_latest']

        while current_time < duration:
            # 实验阶段：主要访问最新版本
            experiment_duration = random.uniform(10, 30)
            experiment_end = current_time + experiment_duration

            while current_time < experiment_end and current_time < duration:
                # 80%概率访问最新版本，20%概率访问历史版本
                if random.random() < 0.8:
                    version = 'checkpoint_latest'
                else:
                    version = random.choice(versions[:-1])

                requests.append({
                    'id': request_id,
                    'timestamp': current_time,
                    'layer': version,
                    'operation': 'read',
                    'size': random.randint(100 * 1024 * 1024, 500 * 1024 * 1024),  # 100-500MB
                    'pattern': 'experiment_access'
                })
                request_id += 1
                current_time += random.uniform(2, 10)

            # 保存新版本
            if current_time < duration:
                requests.append({
                    'id': request_id,
                    'timestamp': current_time,
                    'layer': f'checkpoint_v{random.randint(1, 10)}',
                    'operation': 'write',
                    'size': random.randint(100 * 1024 * 1024, 500 * 1024 * 1024),
                    'pattern': 'version_save'
                })
                request_id += 1
                current_time += random.uniform(5, 15)

        return requests

    def _get_layer_size(self, layer_name):
        """获取层的典型大小"""
        size_map = {
            'embedding': 4 * 1024 * 1024,  # 4MB
            'layer0': 2 * 1024 * 1024,  # 2MB
            'layer1': 2 * 1024 * 1024,  # 2MB
            'layer2': 2 * 1024 * 1024,  # 2MB
            'output': 1 * 1024 * 1024,  # 1MB
            'checkpoint': 100 * 1024 * 1024  # 100MB
        }
        return size_map.get(layer_name, 1 * 1024 * 1024)  # 默认1MB

    def _update_statistics(self, requests, scenario):
        """更新访问统计"""
        self.access_statistics['total_requests'] += len(requests)

        for request in requests:
            layer = request['layer']
            pattern = request['pattern']

            # 层访问统计
            if layer not in self.access_statistics['layer_access_count']:
                self.access_statistics['layer_access_count'][layer] = 0
            self.access_statistics['layer_access_count'][layer] += 1

            # 模式频率统计
            pattern_key = f"{scenario}_{pattern}"
            if pattern_key not in self.access_statistics['pattern_frequency']:
                self.access_statistics['pattern_frequency'][pattern_key] = 0
            self.access_statistics['pattern_frequency'][pattern_key] += 1

    def get_statistics(self):
        """获取工作负载统计"""
        return self.access_statistics

    def save_workload_to_file(self, workload, filename):
        """保存工作负载到文件"""
        with open(filename, 'w') as f:
            json.dump(workload, f, indent=2)
        logger.info(f"工作负载已保存到: {filename}")

    def load_workload_from_file(self, filename):
        """从文件加载工作负载"""
        with open(filename, 'r') as f:
            workload = json.load(f)
        logger.info(f"工作负载已从 {filename} 加载")
        return workload


# 全局工作负载生成器实例
_workload_generator = None


def get_workload_generator():
    """获取全局工作负载生成器实例"""
    global _workload_generator
    if _workload_generator is None:
        _workload_generator = RealWorkloadGenerator()
    return _workload_generator