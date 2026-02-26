#!/usr/bin/env python3
"""
综合评估器 - 专注于真实性能指标
移除所有评分评级逻辑，专注于真实数据统计
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AAT-Evaluator")


@dataclass
class CostMetrics:
    """成本指标"""
    storage_cost: float  # 存储成本
    transfer_cost: float  # 传输成本
    compute_cost: float  # 计算成本
    total_cost: float  # 总成本


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_latency_ms: float  # 平均延迟
    p95_latency_ms: float  # P95延迟
    p99_latency_ms: float  # P99延迟
    throughput_req_per_sec: float  # 吞吐量
    min_latency_ms: float  # 最小延迟
    max_latency_ms: float  # 最大延迟
    total_requests: int  # 总请求数


@dataclass
class EfficiencyMetrics:
    """效率指标"""
    cache_hit_rate_percent: float  # 缓存命中率
    prefetch_accuracy_percent: float  # 预取准确率
    memory_utilization_percent: float  # 内存利用率
    storage_savings_percent: float  # 存储节省
    bandwidth_savings_percent: float  # 带宽节省
    total_cache_access: int  # 总缓存访问次数


@dataclass
class SystemMetrics:
    """系统指标"""
    total_requests: int
    hot_hits: int
    cold_hits: int
    prefetch_hits: int
    total_data_transferred_bytes: int


class TechnicalEvaluator:
    """技术评估器 - 专注于真实性能指标，移除评分逻辑"""

    def __init__(self):
        self.cost_config = {
            'hot_storage': 0.10,  # $/GB-month (内存)
            'warm_storage': 0.05,  # $/GB-month (SSD)
            'cold_storage': 0.01,  # $/GB-month (对象存储)
            'transfer_cost': 0.09,  # $/GB (数据传输)
            'compute_cost': 0.40  # $/hour (计算资源)
        }

    def evaluate_performance(self, access_times: List[float], throughput: float,
                             total_requests: int) -> PerformanceMetrics:
        """评估性能指标 - 只返回真实数据"""
        if not access_times:
            return PerformanceMetrics(
                avg_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_req_per_sec=0,
                min_latency_ms=0,
                max_latency_ms=0,
                total_requests=0
            )

        times_ms = [t * 1000 for t in access_times]

        return PerformanceMetrics(
            avg_latency_ms=round(np.mean(times_ms), 2),
            p95_latency_ms=round(np.percentile(times_ms, 95), 2),
            p99_latency_ms=round(np.percentile(times_ms, 99), 2),
            throughput_req_per_sec=round(throughput, 2),
            min_latency_ms=round(np.min(times_ms), 2),
            max_latency_ms=round(np.max(times_ms), 2),
            total_requests=total_requests
        )

    def evaluate_cost(self, storage_usage: Dict, data_transfer_gb: float,
                      compute_time_hours: float, duration_hours: float) -> CostMetrics:
        """评估成本指标"""
        # 存储成本
        storage_cost = 0
        for tier, usage_gb in storage_usage.items():
            cost_rate = self.cost_config.get(f'{tier}_storage', 0.01)
            storage_cost += cost_rate * usage_gb * (duration_hours / 720)  # 按小时比例计算

        # 传输成本
        transfer_cost = self.cost_config['transfer_cost'] * data_transfer_gb

        # 计算成本
        compute_cost = self.cost_config['compute_cost'] * compute_time_hours

        total_cost = storage_cost + transfer_cost + compute_cost

        return CostMetrics(
            storage_cost=round(storage_cost, 4),
            transfer_cost=round(transfer_cost, 4),
            compute_cost=round(compute_cost, 4),
            total_cost=round(total_cost, 4)
        )

    def evaluate_efficiency(self, cache_stats: Dict, prefetch_stats: Dict,
                            memory_usage: Dict, original_size: int,
                            compressed_size: int) -> EfficiencyMetrics:
        """评估效率指标"""
        # 缓存命中率
        total_cache_access = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
        cache_hit_rate = cache_stats.get('hits', 0) / total_cache_access if total_cache_access > 0 else 0

        # 预取准确率
        total_prefetch = prefetch_stats.get('hits', 0) + prefetch_stats.get('misses', 0)
        prefetch_accuracy = prefetch_stats.get('hits', 0) / total_prefetch if total_prefetch > 0 else 0

        # 内存利用率
        memory_utilization = memory_usage.get('used', 0) / memory_usage.get('total', 1) * 100

        # 存储节省
        storage_savings = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

        # 带宽节省 (基于预取命中率估算)
        bandwidth_savings = prefetch_accuracy * 100

        return EfficiencyMetrics(
            cache_hit_rate_percent=round(cache_hit_rate * 100, 2),
            prefetch_accuracy_percent=round(prefetch_accuracy * 100, 2),
            memory_utilization_percent=round(memory_utilization, 2),
            storage_savings_percent=round(storage_savings, 2),
            bandwidth_savings_percent=round(bandwidth_savings, 2),
            total_cache_access=total_cache_access
        )

    def evaluate_system_metrics(self, performance_stats: Dict, total_data_transferred: int) -> SystemMetrics:
        """评估系统指标"""
        return SystemMetrics(
            total_requests=performance_stats.get('total_requests', 0),
            hot_hits=performance_stats.get('hot_hits', 0),
            cold_hits=performance_stats.get('cold_hits', 0),
            prefetch_hits=performance_stats.get('prefetch_hits', 0),
            total_data_transferred_bytes=total_data_transferred
        )

    def generate_comprehensive_report(self, performance: PerformanceMetrics,
                                      cost: CostMetrics, efficiency: EfficiencyMetrics,
                                      system: SystemMetrics, scenario: str) -> Dict:
        """生成综合评估报告 - 只包含真实数据"""

        report = {
            'scenario': scenario,
            'timestamp': time.time(),
            'performance_metrics': {
                '平均延迟_ms': performance.avg_latency_ms,
                'P95延迟_ms': performance.p95_latency_ms,
                'P99延迟_ms': performance.p99_latency_ms,
                '吞吐量_req_per_sec': performance.throughput_req_per_sec,
                '最小延迟_ms': performance.min_latency_ms,
                '最大延迟_ms': performance.max_latency_ms,
                '总请求数': performance.total_requests
            },
            'cost_metrics': {
                '存储成本_$': cost.storage_cost,
                '传输成本_$': cost.transfer_cost,
                '计算成本_$': cost.compute_cost,
                '总成本_$': cost.total_cost
            },
            'efficiency_metrics': {
                '缓存命中率_%': efficiency.cache_hit_rate_percent,
                '预取准确率_%': efficiency.prefetch_accuracy_percent,
                '内存利用率_%': efficiency.memory_utilization_percent,
                '存储节省_%': efficiency.storage_savings_percent,
                '带宽节省_%': efficiency.bandwidth_savings_percent,
                '总缓存访问次数': efficiency.total_cache_access
            },
            'system_metrics': {
                '总请求数': system.total_requests,
                '热层命中数': system.hot_hits,
                '冷层命中数': system.cold_hits,
                '预取命中数': system.prefetch_hits,
                '数据传输量_bytes': system.total_data_transferred_bytes
            }
        }

        return report

    def compare_scenarios(self, scenario_reports: Dict[str, Dict]) -> Dict:
        """对比不同场景的性能"""
        comparison = {}

        for scenario_name, report in scenario_reports.items():
            comparison[scenario_name] = {
                '平均延迟_ms': report['performance_metrics']['平均延迟_ms'],
                '吞吐量_req_per_sec': report['performance_metrics']['吞吐量_req_per_sec'],
                '缓存命中率_%': report['efficiency_metrics']['缓存命中率_%'],
                '总成本_$': report['cost_metrics']['总成本_$']
            }

        return comparison

    def calculate_data_throughput(self, total_data_bytes: int, total_time_seconds: float) -> float:
        """计算数据吞吐量"""
        if total_time_seconds <= 0:
            return 0
        return round(total_data_bytes / total_time_seconds / 1024 / 1024, 2)  # MB/s

    def calculate_request_distribution(self, access_times: List[float]) -> Dict:
        """计算请求延迟分布"""
        if not access_times:
            return {}

        times_ms = [t * 1000 for t in access_times]

        return {
            '延迟分布_ms': {
                '<10ms': len([t for t in times_ms if t < 10]),
                '10-50ms': len([t for t in times_ms if 10 <= t < 50]),
                '50-100ms': len([t for t in times_ms if 50 <= t < 100]),
                '100-500ms': len([t for t in times_ms if 100 <= t < 500]),
                '>=500ms': len([t for t in times_ms if t >= 500])
            },
            '延迟统计_ms': {
                '标准差': round(np.std(times_ms), 2),
                '中位数': round(np.median(times_ms), 2)
            }
        }


# 向后兼容的简化版本
class ComprehensiveEvaluator:
    """综合评估器 - 简化版本，移除业务指标"""

    def __init__(self):
        self.technical_evaluator = TechnicalEvaluator()

    def evaluate_performance(self, access_times, throughput, total_requests):
        return self.technical_evaluator.evaluate_performance(access_times, throughput, total_requests)

    def evaluate_cost(self, storage_usage, data_transfer, compute_time, duration):
        return self.technical_evaluator.evaluate_cost(storage_usage, data_transfer, compute_time, duration)

    def evaluate_efficiency(self, cache_stats, prefetch_stats, memory_usage, original_size, compressed_size):
        return self.technical_evaluator.evaluate_efficiency(cache_stats, prefetch_stats, memory_usage, original_size,
                                                            compressed_size)


if __name__ == "__main__":
    # 测试评估器
    evaluator = TechnicalEvaluator()

    # 测试数据
    access_times = [0.001, 0.002, 0.0015, 0.003, 0.0025]
    throughput = 100.5
    total_requests = 1000

    performance = evaluator.evaluate_performance(access_times, throughput, total_requests)
    cost = evaluator.evaluate_cost({'hot': 0.5, 'cold': 2.0}, 1.5, 0.5, 1.0)
    efficiency = evaluator.evaluate_efficiency(
        {'hits': 80, 'misses': 20},
        {'hits': 15, 'misses': 5},
        {'used': 0.8, 'total': 1.0},
        100 * 1024 * 1024,
        60 * 1024 * 1024
    )
    system = evaluator.evaluate_system_metrics(
        {'total_requests': 1000, 'hot_hits': 800, 'cold_hits': 200, 'prefetch_hits': 150},
        500 * 1024 * 1024
    )

    report = evaluator.generate_comprehensive_report(performance, cost, efficiency, system, 'online_inference')

    print("评估器测试结果:")
    print(f"性能: {report['performance_metrics']}")
    print(f"成本: {report['cost_metrics']}")
    print(f"效率: {report['efficiency_metrics']}")