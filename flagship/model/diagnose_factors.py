"""
因子诊断脚本：验证“因子与因子关系（共线性/独立性）”，并可选用 LLM 生成总结。

输出（HTML）包含：
- 因子-因子 Spearman 相关性热力图（默认：逐交易日横截面相关 → 对相关矩阵取均值）
- Top correlated pairs（|corr| >= threshold）
- （可选）LightGBM feature importance（gain）
- （可选）LLM 总结（使用 vt_setting.json 或环境变量的 OpenAI API Key）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import Segment
from vnpy.trader.logger import logger


DIAGNOSTICS_SUMMARY_SYSTEM_PROMPT = """你是资深量化研究员。请基于输入的因子诊断数据，用中文输出一段可直接放入研究报告的总结。

要求：
- 聚焦“因子与因子关系（共线性/重合度）”及其对模型与策略研究的影响。
- 必须给出：总体结论、最严重的共线性簇/因子对、对特征工程/删减的具体建议、下一步验证动作。
- 不要输出任何 API key 或敏感配置值。
- 输出格式使用 Markdown（含小标题和项目符号）。
"""


@dataclass(frozen=True)
class DiagnosticsConfig:
    lab_path: Path
    dataset_name: str
    model_name: str
    segment: str
    label_column: str
    max_rows: int
    corr_mode: str
    min_cross_section: int
    corr_threshold: float
    top_k_importance: int
    output_path: Path
    seed: int
    llm_summary: bool
    llm_model: str
    llm_timeout: int
    llm_max_completion_tokens: int


def parse_args() -> DiagnosticsConfig:
    parser = argparse.ArgumentParser(description="Factor diagnostics (correlation + importance + LLM summary)")
    parser.add_argument("--lab-path", type=str, default="lab/flagship_alpha_momentum")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--segment",
        type=str,
        choices=["train", "valid", "test", "all"],
        default="train",
        help="用于诊断的数据段（默认 train）",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="rank_5d",
        help="标签列名（用于排除，不参与相关性计算）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="最大采样行数（防止相关性计算过慢）",
    )
    parser.add_argument(
        "--corr-mode",
        type=str,
        choices=["cross_sectional_mean", "pooled"],
        default="cross_sectional_mean",
        help="相关性口径：cross_sectional_mean=逐交易日横截面相关后取均值；pooled=全样本混合相关",
    )
    parser.add_argument(
        "--min-cross-section",
        type=int,
        default=20,
        help="横截面相关的最小样本数（每个交易日最少 vt_symbol 行数）",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.9,
        help="高相关阈值（用于输出 Top correlated pairs）",
    )
    parser.add_argument(
        "--top-k-importance",
        type=int,
        default=30,
        help="展示 Top-K feature importance（gain，默认 30）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="输出 HTML 路径（默认 <lab_path>/report/model_diagnostics.html）",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--llm-summary",
        action="store_true",
        default=False,
        help="启用 OpenAI LLM 总结（需要在 vt_setting.json 或环境变量中提供 api_key）",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.2",
        help="用于总结的 OpenAI 模型名（默认 gpt-5.2）",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=40,
        help="LLM 请求超时（秒）",
    )
    parser.add_argument(
        "--llm-max-completion-tokens",
        type=int,
        default=2000,
        help="LLM 总结最大 completion tokens（gpt-5.2 在复杂输入下可能需要 >=1500 才会输出内容）",
    )

    args = parser.parse_args()

    lab_path = Path(args.lab_path)
    output_path = Path(args.output_path) if args.output_path else lab_path / "report" / "model_diagnostics.html"

    return DiagnosticsConfig(
        lab_path=lab_path,
        dataset_name=str(args.dataset_name),
        model_name=str(args.model_name),
        segment=str(args.segment),
        label_column=str(args.label_column),
        max_rows=int(args.max_rows),
        corr_mode=str(args.corr_mode),
        min_cross_section=int(args.min_cross_section),
        corr_threshold=float(args.corr_threshold),
        top_k_importance=int(args.top_k_importance),
        output_path=output_path,
        seed=int(args.seed),
        llm_summary=bool(args.llm_summary),
        llm_model=str(args.llm_model),
        llm_timeout=int(args.llm_timeout),
        llm_max_completion_tokens=int(args.llm_max_completion_tokens),
    )


def _load_learn_df(dataset: Any, segment: str) -> pl.DataFrame:
    if segment == "all":
        df = getattr(dataset, "learn_df", None)
        if df is None:
            raise RuntimeError("dataset.learn_df 不存在，无法使用 segment=all")
        return df
    seg = Segment[segment.upper()]
    return dataset.fetch_learn(seg)


def _infer_label_column(df: pl.DataFrame, preferred: str) -> str | None:
    candidates = [preferred, "label_excess_5d", "rank_5d", "label"]
    for c in candidates:
        if c and c in df.columns:
            return c
    return None


def _numeric_feature_columns(df: pl.DataFrame, *, label_col: str | None) -> list[str]:
    exclude = {"datetime", "vt_symbol", "label"}
    if label_col:
        exclude.add(label_col)

    feature_cols = [c for c in df.columns if c not in exclude]
    numeric_cols: list[str] = []
    for c in feature_cols:
        dtype = df.schema.get(c)
        if dtype is None:
            continue
        if hasattr(dtype, "is_numeric") and dtype.is_numeric():  # type: ignore[attr-defined]
            numeric_cols.append(c)
    return numeric_cols


def _spearman_corr_pooled(df: pl.DataFrame, cols: list[str]) -> tuple[list[str], Any]:
    """全样本混合相关（跨日期+跨股票），用于快速粗略检查。"""
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pandas 不可用，无法计算相关性: {exc}") from exc

    pd_df = df.select(cols).to_pandas()
    corr_df = pd_df.corr(method="spearman")
    return cols, corr_df


def _spearman_corr_cross_sectional_mean(
    df: pl.DataFrame,
    cols: list[str],
    *,
    min_cross_section: int,
) -> tuple[list[str], Any, int]:
    """
    逐交易日（datetime）做横截面相关（跨 vt_symbol），再对相关矩阵取均值。
    返回：cols, mean_corr_df(pandas), used_dates_count
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"numpy/pandas 不可用，无法计算横截面相关性: {exc}") from exc

    pd_df = df.select(["datetime", "vt_symbol", *cols]).to_pandas()

    used = 0
    acc: Any | None = None
    for _, g in pd_df.groupby("datetime"):
        if g.shape[0] < min_cross_section:
            continue
        corr = g[cols].corr(method="spearman").to_numpy()
        acc = corr if acc is None else (acc + corr)
        used += 1

    if acc is None or used == 0:
        raise RuntimeError("横截面样本不足：没有任何交易日满足最小横截面样本数要求")

    mean_corr = acc / float(used)
    corr_df = pd.DataFrame(mean_corr, index=cols, columns=cols)
    return cols, corr_df, used


def _top_correlated_pairs(cols: list[str], corr_df: Any, *, threshold: float, top_k: int = 20) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr_df.iat[i, j])
            if abs(v) >= threshold:
                pairs.append((cols[i], cols[j], v))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_k]


def _load_lgb_importance(lab: AlphaLab, model_name: str) -> pl.DataFrame | None:
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as exc:
        logger.warning(f"[diagnose_factors] lightgbm 不可用，跳过 Feature Importance: {exc}")
        return None

    model = lab.load_model(model_name)
    if model is None:
        logger.warning(f"[diagnose_factors] 模型不存在: {model_name}，跳过 Feature Importance")
        return None

    if not isinstance(model, lgb.Booster):
        logger.warning(f"[diagnose_factors] 不支持的模型类型: {type(model)}，跳过 Feature Importance")
        return None

    names = model.feature_name()
    if not names:
        logger.warning("[diagnose_factors] Booster feature_name 为空，跳过 Feature Importance")
        return None

    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")

    return (
        pl.DataFrame(
            {
                "feature": names,
                "importance_gain": [float(x) for x in gain],
                "importance_split": [int(x) for x in split],
            }
        )
        .sort("importance_gain", descending=True)
    )


def _load_openai_api_key() -> str | None:
    """
    优先读取环境变量 OPENAI_API_KEY；否则读取项目根目录下的 vt_setting.json（用户要求）；最后回退到 vn.py 的 SETTINGS。
    注意：不要在日志中打印 key。
    """
    import os

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()

    # 用户要求：从项目目录 vt_setting.json 读取（而不是 vn.py 的 temp path）
    try:
        import json

        local_path = Path.cwd() / "vt_setting.json"
        if local_path.exists():
            with open(local_path, encoding="utf-8") as f:
                data = json.load(f)
            key_local = data.get("open-ai.api_key")
            if isinstance(key_local, str) and key_local.strip():
                return key_local.strip()
    except Exception:
        pass

    try:
        from vnpy.trader.setting import SETTINGS
    except Exception:
        return None

    key2 = SETTINGS.get("open-ai.api_key")
    if isinstance(key2, str) and key2.strip():
        return key2.strip()
    return None


def _summarize_with_openai(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    max_completion_tokens: int,
) -> str:
    """Call OpenAI Chat Completions API via stdlib (no extra deps)."""
    import json
    import urllib.request

    user_content = json.dumps(payload, ensure_ascii=False, indent=2)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        # gpt-5.x: chat/completions uses max_completion_tokens (max_tokens is rejected)
        "max_completion_tokens": int(max_completion_tokens),
    }

    req = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return str(data["choices"][0]["message"]["content"] or "").strip()


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_html(
    *,
    title: str,
    subtitle: str,
    corr_html: str,
    importance_html: str,
    top_pairs: list[tuple[str, str, float]],
    meta: dict[str, Any],
    llm_summary_md: str | None,
) -> str:
    pairs_html = "<p>无（阈值较高或特征较独立）</p>\n"
    if top_pairs:
        rows = "\n".join(
            f"<tr><td>{a}</td><td>{b}</td><td style='text-align:right;'>{v:.4f}</td></tr>"
            for a, b, v in top_pairs
        )
        pairs_html = (
            "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse:collapse; width:100%;'>"
            "<tr><th>Factor A</th><th>Factor B</th><th>Spearman Corr</th></tr>"
            f"{rows}</table>\n"
        )

    meta_rows = "\n".join(
        f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
        for k, v in meta.items()
    )

    llm_html = ""
    if llm_summary_md:
        llm_html = (
            "<h2>LLM 总结</h2>\n"
            "<div class='note'><pre style='white-space:pre-wrap; margin:0;'>"
            f"{_escape_html(llm_summary_md)}"
            "</pre></div>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Ubuntu", sans-serif; margin: 20px; line-height: 1.6; }}
    h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
    h2 {{ color: #555; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
    table {{ margin: 12px 0; border-collapse: collapse; width: 100%; }}
    th {{ background-color: #f0f0f0; font-weight: bold; padding: 8px; text-align: left; }}
    td {{ padding: 8px; }}
    .note {{ background-color: #e8f4f8; border-left: 4px solid #2196F3; padding: 12px; margin: 16px 0; }}
    .warn {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p><strong>生成时间</strong>: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  <p><strong>说明</strong>: {subtitle}</p>

  {llm_html}

  <h2>元信息</h2>
  <table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>
    {meta_rows}
  </table>

  <h2>Spearman 相关性热力图</h2>
  {corr_html}

  <h2>Top Correlated Pairs</h2>
  {pairs_html}

  <h2>Feature Importance（LightGBM）</h2>
  {importance_html}

</body>
</html>"""


def main() -> None:
    cfg = parse_args()

    lab = AlphaLab(str(cfg.lab_path))
    dataset = lab.load_dataset(cfg.dataset_name)
    if dataset is None:
        raise SystemExit(f"Dataset not found: {cfg.dataset_name}")

    df = _load_learn_df(dataset, cfg.segment)
    if df.is_empty():
        raise SystemExit(f"Empty learn df for segment={cfg.segment}")

    # 采样，避免相关性计算过慢（对横截面口径通常不需要太大）
    if cfg.max_rows > 0 and df.height > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, seed=cfg.seed)

    label_col = _infer_label_column(df, cfg.label_column)
    factor_cols = _numeric_feature_columns(df, label_col=label_col)
    if not factor_cols:
        raise SystemExit("No numeric factor columns found.")

    # Correlation
    used_dates: int | None = None
    if cfg.corr_mode == "cross_sectional_mean":
        cols, corr_df, used_dates = _spearman_corr_cross_sectional_mean(
            df,
            factor_cols,
            min_cross_section=cfg.min_cross_section,
        )
    else:
        cols, corr_df = _spearman_corr_pooled(df, factor_cols)

    top_pairs = _top_correlated_pairs(cols, corr_df, threshold=cfg.corr_threshold, top_k=20)

    # Plotly figures
    import plotly.graph_objects as go

    corr_fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=cols,
            y=cols,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Spearman"),
        )
    )
    corr_fig.update_layout(
        height=900,
        width=1100,
        title=f"Spearman Correlation Heatmap ({cfg.dataset_name}, {cfg.segment}, mode={cfg.corr_mode})",
    )
    corr_html = corr_fig.to_html(include_plotlyjs="cdn", full_html=False)

    # Importance (optional)
    importance_df = _load_lgb_importance(lab, cfg.model_name)
    if importance_df is not None and not importance_df.is_empty():
        top_imp = importance_df.head(cfg.top_k_importance)
        imp_fig = go.Figure(
            data=go.Bar(
                x=top_imp["importance_gain"].to_list(),
                y=top_imp["feature"].to_list(),
                orientation="h",
                name="gain",
            )
        )
        imp_fig.update_layout(
            height=max(500, 18 * cfg.top_k_importance),
            width=1100,
            title=f"Top {cfg.top_k_importance} Feature Importance (gain)",
            yaxis=dict(autorange="reversed"),
        )
        importance_html = imp_fig.to_html(include_plotlyjs=False, full_html=False)
    else:
        importance_html = "<p>模型不可用或未加载成功，已跳过 Feature Importance。</p>"

    # LLM summary (required if enabled)
    llm_summary_md: str | None = None
    if cfg.llm_summary:
        api_key = _load_openai_api_key()
        if not api_key:
            raise SystemExit("OpenAI API key missing: set OPENAI_API_KEY or vt_setting.json: open-ai.api_key")

        payload = {
            "dataset_name": cfg.dataset_name,
            "model_name": cfg.model_name,
            "segment": cfg.segment,
            "corr_mode": cfg.corr_mode,
            "min_cross_section": cfg.min_cross_section,
            "rows_used": int(df.height),
            "factors_count": len(cols),
            "dates_used": int(used_dates) if used_dates is not None else None,
            "corr_threshold": cfg.corr_threshold,
            "top_correlated_pairs": [
                {"a": a, "b": b, "corr": float(v)} for a, b, v in top_pairs
            ],
            "top_importance_gain": (
                [
                    {"feature": r["feature"], "gain": float(r["importance_gain"])}
                    for r in (importance_df.head(cfg.top_k_importance).to_dicts() if importance_df is not None else [])
                ]
            ),
        }
        llm_summary_md = _summarize_with_openai(
            api_key=api_key,
            model=cfg.llm_model,
            system_prompt=DIAGNOSTICS_SUMMARY_SYSTEM_PROMPT,
            payload=payload,
            timeout_seconds=cfg.llm_timeout,
            max_completion_tokens=cfg.llm_max_completion_tokens,
        )
        # gpt-5.2 在复杂输入下可能出现“content 为空但 finish_reason=length”，这里兜底重试一次
        if not llm_summary_md:
            retry_tokens = min(max(cfg.llm_max_completion_tokens * 2, cfg.llm_max_completion_tokens + 1000), 5000)
            logger.warning(
                f"[diagnose_factors] LLM summary empty, retry once with max_completion_tokens={retry_tokens}"
            )
            llm_summary_md = _summarize_with_openai(
                api_key=api_key,
                model=cfg.llm_model,
                system_prompt=DIAGNOSTICS_SUMMARY_SYSTEM_PROMPT,
                payload=payload,
                timeout_seconds=cfg.llm_timeout,
                max_completion_tokens=retry_tokens,
            )
            if not llm_summary_md:
                raise SystemExit("LLM summary empty after retry; increase --llm-max-completion-tokens")

    meta = {
        "lab_path": str(cfg.lab_path),
        "dataset_name": cfg.dataset_name,
        "model_name": cfg.model_name,
        "segment": cfg.segment,
        "corr_mode": cfg.corr_mode,
        "min_cross_section": cfg.min_cross_section,
        "rows_used": int(df.height),
        "numeric_factors": len(cols),
        "dates_used": int(used_dates) if used_dates is not None else "N/A",
        "corr_threshold": cfg.corr_threshold,
        "top_pairs_found": len(top_pairs),
        "llm_summary": cfg.llm_summary,
        "llm_model": cfg.llm_model if cfg.llm_summary else "N/A",
        "llm_max_completion_tokens": cfg.llm_max_completion_tokens if cfg.llm_summary else "N/A",
    }

    html = _build_html(
        title="FCAM 因子诊断报告（相关性/重要性）",
        subtitle="用于验证因子共线性（独立性）与模型依赖的特征重要性。",
        corr_html=corr_html,
        importance_html=importance_html,
        top_pairs=top_pairs,
        meta=meta,
        llm_summary_md=llm_summary_md,
    )

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(html, encoding="utf-8")
    logger.info(f"[diagnose_factors] saved: {cfg.output_path}")


if __name__ == "__main__":
    main()


