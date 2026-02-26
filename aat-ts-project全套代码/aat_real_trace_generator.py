#!/usr/bin/env python3
"""
真实AI工作负载Trace生成器 - 修复版
确保生成有意义的请求序列，避免空trace
"""

import json
import time
import random
import numpy as np
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AAT-Trace-Generator")


class RealAITraceGenerator:
    def __init__(self):
        # 基于真实BERT推理trace分析的模式
        self.trace_patterns = {
            'online_inference': self._generate_online_trace,
            'batch_inference': self._generate_batch_trace,
            'edge_finetuning': self._generate_edge_trace
        }

        # 真实trace统计特征
        self.trace_stats = {
            'request_sizes': [512, 1024, 2048, 4096, 8192],
            'layer_access_sequences': [
                ['embedding', 'layer0', 'output'],
                ['embedding', 'layer0', 'layer1', 'output'],
                ['embedding', 'layer0', 'layer1', 'layer2', 'output'],
                ['embedding', 'layer0', 'layer1', 'layer2', 'layer3', 'output']
            ],
            'qps_distribution': {
                'low': {'mean': 5, 'std': 2},
                'medium': {'mean': 20, 'std': 5},
                'high': {'mean': 100, 'std': 20}
            }
        }

    def generate_trace(self, scenario, duration=60, intensity='medium', output_file=None):
        """生成指定场景的trace - 修复空请求问题"""
        if scenario not in self.trace_patterns:
            raise ValueError(f"不支持的场景: {scenario}")

        logger.info(f"生成 {scenario} trace，强度: {intensity}，时长: {duration}s")

        trace = {
            'scenario': scenario,
            'intensity': intensity,
            'duration': duration,
            'start_time': time.time(),
            'requests': []
        }

        # 生成请求序列
        requests = self.trace_patterns[scenario](duration, intensity)

        # 如果没有生成请求，创建基础请求序列
        if not requests:
            logger.warning("未生成请求，创建基础请求序列")
            requests = self._create_basic_requests(scenario, duration, intensity)

        trace['requests'] = requests

        # 保存到文件
        if output_file:
            self.save_trace(trace, output_file)

        return trace

    def _create_basic_requests(self, scenario, duration, intensity):
        """创建基础请求序列 - 确保总有请求生成"""
        requests = []
        current_time = 0
        request_id = 0

        # 根据场景和强度确定请求数量
        intensity_params = {
            'low': {'requests_per_sec': 2, 'patterns': 2},
            'medium': {'requests_per_sec': 5, 'patterns': 3},
            'high': {'requests_per_sec': 10, 'patterns': 4}
        }

        params = intensity_params.get(intensity, intensity_params['medium'])
        total_requests = int(params['requests_per_sec'] * duration)

        # 根据场景选择layer序列
        if scenario == 'online_inference':
            base_patterns = [
                ['embedding', 'layer0', 'output'],
                ['embedding', 'layer0', 'layer1', 'output'],
                ['embedding', 'layer0', 'layer1', 'layer2', 'output']
            ]
        elif scenario == 'edge_finetuning':
            base_patterns = [
                ['embedding', 'layer0', 'layer1'],
                ['embedding', 'layer0', 'layer1', 'checkpoint'],
                ['config', 'embedding', 'layer0', 'layer1', 'checkpoint']
            ]
        else:  # batch_inference or default
            base_patterns = [
                ['embedding', 'layer0', 'layer1', 'layer2', 'layer3', 'output'],
                ['embedding', 'layer0', 'layer1', 'layer2', 'output']
            ]

        # 选择模式
        selected_patterns = base_patterns[:params['patterns']]

        # 生成请求
        request_count = 0
        while current_time < duration and request_count < total_requests:
            pattern = random.choice(selected_patterns)
            interval = max(0.1, random.expovariate(params['requests_per_sec']))

            for layer in pattern:
                if request_count >= total_requests:
                    break

                requests.append({
                    'id': request_id,
                    'timestamp': current_time,
                    'layer': layer,
                    'operation': 'read',
                    'size': random.choice(self.trace_stats['request_sizes']),
                    'pattern': scenario,
                    'intensity': intensity
                })
                request_id += 1
                request_count += 1

            current_time += interval

        logger.info(f"生成基础请求序列: {len(requests)} 个请求")
        return requests

    def _generate_online_trace(self, duration, intensity):
        """生成在线推理trace - 修复版"""
        requests = []
        current_time = 0
        request_id = 0

        # 强度参数
        qps_params = self.trace_stats['qps_distribution'][intensity]
        base_qps = max(1, int(np.random.normal(qps_params['mean'], qps_params['std'])))

        # 确保有足够的请求
        min_requests = max(10, int(duration * base_qps * 0.5))

        # 突发流量模式
        burst_intervals = self._generate_burst_pattern(duration)

        while current_time < duration and len(requests) < min_requests * 2:
            # 基础请求间隔
            base_interval = max(0.05, np.random.exponential(1.0 / base_qps))

            # 突发流量叠加
            burst_factor = 1.0
            for burst_start, burst_end, intensity_factor in burst_intervals:
                if burst_start <= current_time <= burst_end:
                    burst_factor = intensity_factor
                    break

            interval = base_interval / burst_factor
            current_time += interval

            # 选择访问模式
            pattern_weights = [0.4, 0.3, 0.2, 0.1]
            pattern = random.choices(
                self.trace_stats['layer_access_sequences'],
                weights=pattern_weights
            )[0]

            # 请求大小分布
            size_weights = [0.3, 0.4, 0.2, 0.08, 0.02]
            request_size = random.choices(
                self.trace_stats['request_sizes'],
                weights=size_weights
            )[0]

            # 生成请求序列
            for i, layer in enumerate(pattern):
                request_time = current_time + i * 0.001

                requests.append({
                    'id': request_id,
                    'timestamp': request_time,
                    'layer': layer,
                    'operation': 'read',
                    'size': request_size,
                    'pattern': 'online_inference',
                    'burst_level': burst_factor
                })
                request_id += 1

        # 按时间排序
        requests.sort(key=lambda x: x['timestamp'])
        logger.info(f"在线推理trace生成: {len(requests)} 请求")
        return requests

    def _generate_batch_trace(self, duration, intensity):
        """生成批量推理trace - 修复版"""
        requests = []
        current_time = 0
        request_id = 0

        # 批量大小分布
        batch_sizes = [4, 8, 16, 32]
        batch_weights = [0.2, 0.4, 0.3, 0.1]

        min_batches = max(3, int(duration / 10))

        batch_count = 0
        while current_time < duration and batch_count < min_batches:
            # 批量处理间隔
            batch_interval = random.uniform(5.0, 15.0)
            current_time += batch_interval

            # 批量大小
            batch_size = random.choices(batch_sizes, weights=batch_weights)[0]

            # 完整模型推理 - 使用较长的推理路径
            inference_pattern = self.trace_stats['layer_access_sequences'][-1]
            for layer in inference_pattern:
                for batch_idx in range(batch_size):
                    request_time = current_time + batch_idx * 0.02

                    requests.append({
                        'id': request_id,
                        'timestamp': request_time,
                        'layer': layer,
                        'operation': 'read',
                        'size': 4096,
                        'pattern': 'batch_inference',
                        'batch_size': batch_size,
                        'batch_index': batch_idx
                    })
                    request_id += 1

            batch_count += 1

        requests.sort(key=lambda x: x['timestamp'])
        logger.info(f"批量推理trace生成: {len(requests)} 请求")
        return requests

    def _generate_edge_trace(self, duration, intensity):
        """生成边缘微调trace - 修复版"""
        requests = []
        current_time = 0
        request_id = 0

        # 训练周期模式
        training_cycles = max(2, int(duration / 20))
        min_requests = max(20, int(duration * 2))

        for cycle in range(training_cycles):
            # 训练阶段
            training_duration = random.uniform(10, 15)
            training_end = current_time + training_duration

            step = 0
            while current_time < training_end and current_time < duration:
                # 每个训练step访问核心层
                training_layers = ['embedding', 'layer0', 'layer1', 'output']
                for layer in training_layers:
                    requests.append({
                        'id': request_id,
                        'timestamp': current_time,
                        'layer': layer,
                        'operation': 'read',
                        'size': 8192,
                        'pattern': 'training_step',
                        'cycle': cycle,
                        'step': step
                    })
                    request_id += 1
                    current_time += 0.1  # 训练step间隔

                step += 1

            # 保存检查点
            if current_time < duration:
                requests.append({
                    'id': request_id,
                    'timestamp': current_time,
                    'layer': 'checkpoint',
                    'operation': 'write',
                    'size': 10 * 1024 * 1024,
                    'pattern': 'checkpoint_save',
                    'cycle': cycle
                })
                request_id += 1
                current_time += 1.0  # 检查点保存时间

            # 确保有足够的请求
            if len(requests) >= min_requests:
                break

        requests.sort(key=lambda x: x['timestamp'])
        logger.info(f"边缘微调trace生成: {len(requests)} 请求")
        return requests

    def _generate_burst_pattern(self, duration):
        """生成突发流量模式"""
        bursts = []
        num_bursts = max(1, int(duration / 60))

        for i in range(num_bursts):
            burst_start = random.uniform(0, duration * 0.8)
            burst_duration = random.uniform(5, 15)
            burst_intensity = random.choice([2.0, 3.0, 5.0])

            bursts.append((burst_start, burst_start + burst_duration, burst_intensity))

        return bursts

    def save_trace(self, trace, filename):
        """保存trace到文件"""
        with open(filename, 'w') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        logger.info(f"Trace已保存: {filename}")

    def load_trace(self, filename):
        """从文件加载trace"""
        with open(filename, 'r') as f:
            trace = json.load(f)
        logger.info(f"Trace已加载: {filename}")
        return trace

    def analyze_trace(self, trace):
        """分析trace特征"""
        requests = trace['requests']

        analysis = {
            'total_requests': len(requests),
            'duration': trace['duration'],
            'qps': len(requests) / trace['duration'] if trace['duration'] > 0 else 0,
            'layer_distribution': defaultdict(int),
            'size_distribution': defaultdict(int),
            'pattern_distribution': defaultdict(int)
        }

        for req in requests:
            analysis['layer_distribution'][req['layer']] += 1
            analysis['size_distribution'][req['size']] += 1
            analysis['pattern_distribution'][req.get('pattern', 'unknown')] += 1

        return analysis


# 全局trace生成器实例
_trace_generator = None


def get_trace_generator():
    global _trace_generator
    if _trace_generator is None:
        _trace_generator = RealAITraceGenerator()
    return _trace_generator


if __name__ == "__main__":
    # 测试trace生成器
    generator = RealAITraceGenerator()

    print("测试Trace生成器:")
    for scenario in ['online_inference', 'edge_finetuning', 'batch_inference']:
        trace = generator.generate_trace(scenario, duration=10, intensity='medium')
        analysis = generator.analyze_trace(trace)
        print(f"{scenario}: {analysis['total_requests']} 请求, QPS: {analysis['qps']:.2f}")