#!/usr/bin/env python3
import time
import logging
from collections import defaultdict, deque

logger = logging.getLogger("AAT-PerformanceMonitor")


class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.access_times = deque(maxlen=window_size)
        self.layer_access_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.prefetch_stats = {'attempts': 0, 'success': 0}

    def record_access(self, filename, access_time, source):
        """记录访问性能"""
        self.access_times.append(access_time)
        self.layer_access_stats[filename]['count'] += 1
        self.layer_access_stats[filename]['total_time'] += access_time

        if source == 'cache':
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1

    def record_prefetch(self, success=True):
        """记录预取性能"""
        self.prefetch_stats['attempts'] += 1
        if success:
            self.prefetch_stats['success'] += 1

    def get_performance_report(self):
        """生成性能报告"""
        if not self.access_times:
            return {}

        times = list(self.access_times)
        total_accesses = self.cache_stats['hits'] + self.cache_stats['misses']

        report = {
            'avg_access_time': sum(times) / len(times),
            'p95_access_time': sorted(times)[int(0.95 * len(times))] if len(times) > 0 else 0,
            'p99_access_time': sorted(times)[int(0.99 * len(times))] if len(times) > 0 else 0,
            'cache_hit_rate': self.cache_stats['hits'] / total_accesses if total_accesses > 0 else 0,
            'prefetch_success_rate': self.prefetch_stats['success'] / self.prefetch_stats['attempts'] if
            self.prefetch_stats['attempts'] > 0 else 0,
            'total_accesses': total_accesses,
            'layer_stats': {
                layer: {
                    'avg_time': stats['total_time'] / stats['count'] if stats['count'] > 0 else 0,
                    'access_count': stats['count'],
                    'access_frequency': stats['count'] / total_accesses if total_accesses > 0 else 0
                }
                for layer, stats in self.layer_access_stats.items()
            }
        }

        return report

    def clear_stats(self):
        """清空统计"""
        self.access_times.clear()
        self.layer_access_stats.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.prefetch_stats = {'attempts': 0, 'success': 0}
        logger.info("性能统计已清空")