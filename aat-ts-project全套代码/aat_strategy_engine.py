#!/usr/bin/env python3
import logging
import yaml
import os
from dataclasses import dataclass
from enum import Enum
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AAT-Strategy")


class StorageTier(Enum):
    HOT = "hot"  # Redis内存缓存
    WARM = "warm"  # 本地SSD（预留）
    COLD = "cold"  # MinIO对象存储


class OperationMode(Enum):
    PERFORMANCE = "performance"  # 性能优先
    COST_SAVING = "cost_saving"  # 成本优先
    BALANCED = "balanced"  # 平衡模式


@dataclass
class TensorInfo:
    name: str
    size: int
    layer_type: str
    access_frequency: int = 0
    last_access: float = 0


class AdaptiveStrategyEngine:
    def __init__(self, config_path="aat_strategy_config.yaml"):
        self.config = self._load_config(config_path)
        self.tensor_info = {}
        self.current_mode = OperationMode(self.config['default_mode'])

        # 修复访问统计 - 明确的统计逻辑
        self.access_stats = {
            'total_requests': 0,
            'hot_hits': 0,
            'cold_hits': 0,
            'prefetch_hits': 0,
            'last_reset_time': time.time()
        }

        logger.info(f"策略引擎初始化完成，模式: {self.current_mode.value}")

    def _load_config(self, config_path):
        """加载策略配置"""
        default_config = {
            'default_mode': 'performance',
            'tier_selection': {
                'embedding': 'hot',
                'layer0': 'hot',
                'layer1': 'hot',
                'layer2': 'warm',
                'layer3': 'cold',
                'output': 'hot',
                'config': 'cold',
                'checkpoint': 'cold'
            },
            'compression': {
                'enabled': True,
                'min_size': 1024,  # 1KB以上才压缩
                'algorithm': 'gzip'
            },
            'cache_ttl': {
                'hot': 300,  # 5分钟
                'warm': 1800,  # 30分钟
                'cold': 3600  # 1小时
            },
            'performance_weights': {
                'access_frequency': 0.6,
                'tensor_size': 0.2,
                'layer_importance': 0.2
            }
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # 合并配置
                    self._merge_configs(default_config, user_config)
                logger.info(f"已加载配置文件: {config_path}")
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")

        return default_config

    def _merge_configs(self, default, user):
        """递归合并配置"""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_configs(default[key], value)
                else:
                    default[key] = value

    def select_storage_tier(self, filename, size=0, operation="read"):
        """为文件选择存储层级"""
        # 注意：这里只增加总请求数，不重复计数
        # 实际的命中统计在 record_cache_hit 中处理

        # 分类文件类型
        file_type = self._classify_file_type(filename)

        # 更新访问信息
        if filename not in self.tensor_info:
            self.tensor_info[filename] = TensorInfo(
                name=filename,
                size=size,
                layer_type=file_type
            )

        tensor_info = self.tensor_info[filename]
        tensor_info.access_frequency += 1
        tensor_info.last_access = time.time()

        # 根据当前模式和文件类型选择层级
        tier = self._make_tier_decision(tensor_info)

        logger.debug(f"层级选择: {filename} -> {tier.value}")
        return tier

    def _classify_file_type(self, filename):
        """分类文件类型"""
        filename_lower = filename.lower()

        if any(x in filename_lower for x in ['embedding', 'emb']):
            return 'embedding'
        elif any(x in filename_lower for x in ['output', 'head', 'classifier']):
            return 'output'
        elif any(x in filename_lower for x in ['config', 'json', 'yaml']):
            return 'config'
        elif any(x in filename_lower for x in ['checkpoint', 'ckpt']):
            return 'checkpoint'
        elif 'layer0' in filename_lower:
            return 'layer0'
        elif 'layer1' in filename_lower:
            return 'layer1'
        elif 'layer2' in filename_lower:
            return 'layer2'
        elif 'layer3' in filename_lower:
            return 'layer3'
        else:
            return 'other'

    def _make_tier_decision(self, tensor_info):
        """基于策略做出层级决策"""
        # 基础层级选择
        base_tier = self.config['tier_selection'].get(
            tensor_info.layer_type, 'cold'
        )

        # 根据模式调整
        if self.current_mode == OperationMode.PERFORMANCE:
            # 性能模式：更倾向于热层
            if base_tier == 'cold' and tensor_info.access_frequency > 5:
                return StorageTier.HOT
            elif base_tier == 'warm' and tensor_info.access_frequency > 2:
                return StorageTier.HOT

        elif self.current_mode == OperationMode.COST_SAVING:
            # 成本模式：更倾向于冷层
            if base_tier == 'hot' and tensor_info.access_frequency < 3:
                return StorageTier.COLD
            elif base_tier == 'warm' and tensor_info.access_frequency < 2:
                return StorageTier.COLD

        return StorageTier(base_tier)

    def should_compress(self, filename, size):
        """判断是否应该压缩"""
        if not self.config['compression']['enabled']:
            return False

        min_size = self.config['compression']['min_size']
        file_type = self._classify_file_type(filename)

        # 配置层不压缩
        if file_type == 'config':
            return False

        return size >= min_size

    def get_cache_ttl(self, tier):
        """获取缓存TTL"""
        return self.config['cache_ttl'].get(tier.value, 300)

    def set_operation_mode(self, mode):
        """设置操作模式"""
        if isinstance(mode, str):
            mode = OperationMode(mode)

        self.current_mode = mode
        logger.info(f"操作模式已切换: {mode.value}")

    def record_cache_hit(self, tier, prefetched=False):
        """记录缓存命中 - 修复预取统计逻辑"""
        # 增加总请求数
        self.access_stats['total_requests'] += 1

        if tier == StorageTier.HOT:
            self.access_stats['hot_hits'] += 1
            if prefetched:
                self.access_stats['prefetch_hits'] += 1
                logger.debug(f"📊 记录预取命中: {prefetched}")
        elif tier == StorageTier.COLD:
            self.access_stats['cold_hits'] += 1

        logger.debug(f"统计更新: 总请求={self.access_stats['total_requests']}, "
                     f"热命中={self.access_stats['hot_hits']}, "
                     f"预取命中={self.access_stats['prefetch_hits']}")

    def get_performance_stats(self):
        """获取性能统计 - 修复计算逻辑"""
        total = self.access_stats['total_requests']
        if total == 0:
            return {
                'total_requests': 0,
                'hot_hit_rate': 0,
                'prefetch_hit_rate': 0,
                'cold_hit_rate': 0,
                'stats_since': self.access_stats['last_reset_time'],
                'hot_hits': 0,
                'prefetch_hits': 0,
                'cold_hits': 0
            }

        # 确保命中率计算正确
        hot_hit_rate = self.access_stats['hot_hits'] / total
        prefetch_hit_rate = self.access_stats['prefetch_hits'] / total
        cold_hit_rate = self.access_stats['cold_hits'] / total

        return {
            'total_requests': total,
            'hot_hit_rate': hot_hit_rate,
            'prefetch_hit_rate': prefetch_hit_rate,
            'cold_hit_rate': cold_hit_rate,
            'stats_since': self.access_stats['last_reset_time'],
            'hot_hits': self.access_stats['hot_hits'],
            'prefetch_hits': self.access_stats['prefetch_hits'],
            'cold_hits': self.access_stats['cold_hits']
        }

    def reset_stats(self):
        """重置统计"""
        self.access_stats = {
            'total_requests': 0,
            'hot_hits': 0,
            'cold_hits': 0,
            'prefetch_hits': 0,
            'last_reset_time': time.time()
        }
        logger.info("性能统计已重置")

    def save_config(self, config_path="aat_strategy_config.yaml"):
        """保存当前配置"""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"配置已保存: {config_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")