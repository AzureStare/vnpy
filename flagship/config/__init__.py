"""
Flagship 策略通用配置模块。

提供统一的数据源配置、API 客户端创建、路径管理等功能。
"""

from .paths import (
    FlagshipPaths,
    get_paths,
    PROJECT_ROOT,
    VT_SETTING_PATH,
    DEFAULT_UNIVERSE_DIR,
    DEFAULT_LAB_DIR,
)
from .polygon_config import (
    get_polygon_api_key,
    create_polygon_client,
    PolygonConfigError,
)

__all__ = [
    # 路径配置
    "FlagshipPaths",
    "get_paths",
    "PROJECT_ROOT",
    "VT_SETTING_PATH",
    "DEFAULT_UNIVERSE_DIR",
    "DEFAULT_LAB_DIR",
    # Polygon 配置
    "get_polygon_api_key",
    "create_polygon_client",
    "PolygonConfigError",
]

