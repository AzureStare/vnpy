"""
将 Markdown 文档导出为支持 LaTeX 数学公式的 HTML/PDF。

使用 Pandoc 进行转换，支持完整的 Markdown 语法和 LaTeX 数学公式。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

PROJECT_ROOT = _project_root


def check_pandoc() -> bool:
    """检查系统是否安装了 pandoc。"""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_markdown_to_html(
    input_path: Path,
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    使用 Pandoc 将 Markdown 转换为 HTML（支持 LaTeX 数学公式）。

    Args:
        input_path: 输入的 Markdown 文件路径
        output_path: 输出的 HTML 文件路径
        title: HTML 文档标题（可选）
    """
    if not check_pandoc():
        raise RuntimeError(
            "Pandoc 未安装。请访问 https://pandoc.org/installing.html 安装。\n"
            "macOS: brew install pandoc\n"
            "Linux: sudo apt-get install pandoc\n"
            "Windows: choco install pandoc"
        )

    cmd = [
        "pandoc",
        str(input_path),
        "-o",
        str(output_path),
        "--standalone",  # 生成完整的 HTML 文档
        "--mathjax",  # 使用 MathJax 渲染数学公式
        "--toc",  # 生成目录
        "--toc-depth=3",  # 目录深度
    ]

    if title:
        cmd.extend(["--metadata", f"title={title}"])

    # 添加自定义 CSS 样式（可选）
    cmd.extend([
        "--css=https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css",
    ])

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Pandoc 转换失败: {result.stderr}")


def convert_markdown_to_pdf(
    input_path: Path,
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    使用 Pandoc 将 Markdown 转换为 PDF（支持 LaTeX 数学公式）。

    Args:
        input_path: 输入的 Markdown 文件路径
        output_path: 输出的 PDF 文件路径
        title: PDF 文档标题（可选）

    注意：需要安装 LaTeX 发行版（如 TeX Live 或 MiKTeX）。
    """
    if not check_pandoc():
        raise RuntimeError(
            "Pandoc 未安装。请访问 https://pandoc.org/installing.html 安装。"
        )

    cmd = [
        "pandoc",
        str(input_path),
        "-o",
        str(output_path),
        "--pdf-engine=xelatex",  # 使用 XeLaTeX 引擎（支持中文和 Unicode）
        "--toc",  # 生成目录
        "--toc-depth=3",
    ]

    if title:
        cmd.extend(["--metadata", f"title={title}"])

    # 设置中文字体（macOS）
    cmd.extend([
        "-V", "CJKmainfont=PingFang SC",
        "-V", "geometry:margin=2cm",
    ])

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr
        if "xelatex" in error_msg.lower() or "latex" in error_msg.lower():
            raise RuntimeError(
                f"PDF 转换失败（可能需要安装 LaTeX）: {error_msg}\n"
                "macOS: brew install --cask mactex\n"
                "Linux: sudo apt-get install texlive-xetex texlive-lang-chinese\n"
                "Windows: 安装 MiKTeX 或 TeX Live"
            )
        raise RuntimeError(f"Pandoc 转换失败: {error_msg}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 Pandoc 将 Markdown 文档导出为 HTML 或 PDF（支持 LaTeX 数学公式）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "flagship" / "docs" / "Flagship Alpha-Momentum.md",
        help="输入的 Markdown 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出的文件路径（根据扩展名自动判断格式：.html 或 .pdf）",
    )
    parser.add_argument(
        "--format",
        choices=["html", "pdf"],
        help="输出格式（如果未指定 --output，则使用此选项）",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="文档标题（可选）",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 确定输出路径和格式
    if args.output:
        output_path = args.output
        output_format = output_path.suffix[1:].lower() if output_path.suffix else None
    elif args.format:
        output_path = input_path.with_suffix(f".{args.format}")
        output_format = args.format
    else:
        # 默认输出 HTML
        output_path = input_path.with_suffix(".html")
        output_format = "html"

    if output_format not in ["html", "pdf"]:
        output_format = "html"
        output_path = input_path.with_suffix(".html")

    title = args.title or input_path.stem

    # 执行转换
    if output_format == "html":
        print(f"正在将 Markdown 转换为 HTML: {input_path} -> {output_path}")
        convert_markdown_to_html(input_path, output_path, title=title)
        print(f"✓ HTML 文件已生成: {output_path}")
    elif output_format == "pdf":
        print(f"正在将 Markdown 转换为 PDF: {input_path} -> {output_path}")
        convert_markdown_to_pdf(input_path, output_path, title=title)
        print(f"✓ PDF 文件已生成: {output_path}")


if __name__ == "__main__":
    main()
