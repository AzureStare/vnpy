"""
Polygon.io API 配置与客户端创建模块。

统一处理 Polygon API key 的加载和 RESTClient 的创建，避免代码重复。
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polygon.rest import RESTClient

from .paths import get_paths


class PolygonConfigError(RuntimeError):
    """Polygon 配置相关错误。"""
    pass


def get_polygon_api_key() -> str:
    """
    从环境变量或 vt_setting.json 读取 Polygon API key。

    优先级：
    1. 环境变量 POLYGON_API_KEY
    2. vt_setting.json 中的 datafeed.password 或 datafeed.token（当 datafeed.name == "polygon" 时）

    Returns:
        Polygon API key 字符串

    Raises:
        PolygonConfigError: 如果无法找到有效的 API key
    """
    # 优先从环境变量读取
    env_key = os.getenv("POLYGON_API_KEY")
    if env_key:
        return env_key

    # 从 vt_setting.json 读取
    paths = get_paths()
    vt_setting_path = paths.vt_setting_path
    
    if vt_setting_path.exists():
        try:
            data = json.loads(vt_setting_path.read_text(encoding="utf-8"))
            if data.get("datafeed.name", "").lower() == "polygon":
                api_key = data.get("datafeed.password") or data.get("datafeed.token")
                if api_key:
                    return api_key
        except (json.JSONDecodeError, KeyError) as exc:
            raise PolygonConfigError(
                f"Failed to parse vt_setting.json: {exc}"
            ) from exc

    raise PolygonConfigError(
        "Polygon API key not found. "
        "Set POLYGON_API_KEY environment variable or configure vt_setting.json "
        f"(expected path: {vt_setting_path})."
    )


def create_polygon_client(api_key: str | None = None) -> "RESTClient":
    """
    创建 Polygon RESTClient 实例。

    Args:
        api_key: 可选的 API key。如果为 None，则自动调用 get_polygon_api_key() 获取。

    Returns:
        Polygon RESTClient 实例

    Raises:
        PolygonConfigError: 如果无法获取有效的 API key
        ImportError: 如果 polygon 包未安装
    """
    try:
        from polygon.rest import RESTClient
    except ImportError as exc:
        raise ImportError(
            "polygon package not installed. "
            "Install it with: pip install polygon-api-client"
        ) from exc

    if api_key is None:
        api_key = get_polygon_api_key()

    return RESTClient(api_key)

