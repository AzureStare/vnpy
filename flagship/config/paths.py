"""
Flagship 策略路径配置模块。

统一管理所有项目路径，避免在各脚本中重复定义。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class FlagshipPaths:
    """
    Flagship 策略项目路径配置类。

    提供统一的路径访问接口，所有路径基于 PROJECT_ROOT 计算。
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """
        初始化路径配置。

        Args:
            project_root: 项目根目录。如果为 None，则自动计算（基于本文件位置）。
        """
        if project_root is None:
            # 计算项目根目录：从 flagship/config/paths.py 向上 2 级到 vnpy 目录
            # flagship/config/paths.py -> flagship/config -> flagship -> vnpy
            project_root = Path(__file__).resolve().parents[2]

        self.project_root: Path = project_root

    @property
    def vt_setting_path(self) -> Path:
        """vt_setting.json 配置文件路径。"""
        return self.project_root / "vt_setting.json"

    @property
    def universe_dir(self) -> Path:
        """每日股票池文件目录。"""
        return self.project_root / "flagship" / "data" / "universe"

    @property
    def lab_dir(self) -> Path:
        """AlphaLab 数据目录（默认 flagship_alpha_momentum）。"""
        return self.project_root / "lab" / "flagship_alpha_momentum"

    @property
    def scripts_dir(self) -> Path:
        """脚本目录。"""
        return self.project_root / "flagship" / "scripts"

    @property
    def backtest_dir(self) -> Path:
        """回测脚本目录。"""
        return self.project_root / "flagship" / "backtest"

    @property
    def factors_dir(self) -> Path:
        """因子计算脚本目录。"""
        return self.project_root / "flagship" / "factors"

    @property
    def data_raw_dir(self) -> Path:
        """原始数据目录。"""
        return self.project_root / "flagship" / "data" / "raw"

    @property
    def data_cleaned_dir(self) -> Path:
        """清洗后数据目录。"""
        return self.project_root / "flagship" / "data" / "cleaned"

    def get_lab_dir(self, lab_name: str = "flagship_alpha_momentum") -> Path:
        """
        获取指定名称的 AlphaLab 数据目录。

        Args:
            lab_name: AlphaLab 任务名称（默认 flagship_alpha_momentum）

        Returns:
            AlphaLab 数据目录路径
        """
        return self.project_root / "lab" / lab_name


# 全局单例实例
_paths_instance: FlagshipPaths | None = None


def get_paths() -> FlagshipPaths:
    """
    获取全局路径配置实例（单例模式）。

    Returns:
        FlagshipPaths 实例
    """
    global _paths_instance
    if _paths_instance is None:
        _paths_instance = FlagshipPaths()
    return _paths_instance


# 向后兼容：导出常用路径作为模块级变量
_paths = get_paths()
PROJECT_ROOT = _paths.project_root
VT_SETTING_PATH = _paths.vt_setting_path
DEFAULT_UNIVERSE_DIR = _paths.universe_dir
DEFAULT_LAB_DIR = _paths.lab_dir

