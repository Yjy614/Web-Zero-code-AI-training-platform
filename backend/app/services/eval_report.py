"""评估报告 HTML/JSON 写入。"""

from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# 北京时间（东八区）
_TZ_BEIJING = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _beijing_now_str() -> str:
    """报告用北京时间，形如 2026-08-14 21:43:40。"""
    return datetime.now(_TZ_BEIJING).strftime("%Y-%m-%d %H:%M:%S")


def write_eval_reports(task_name: str, metrics: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    """
    写入 eval_report.json 与 eval_report.html。
    若 metrics 含 ai_advice，HTML 中优先展示 AI 优化建议。
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "eval_report.json"
    report_html = report_dir / "eval_report.html"

    report_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    map50 = metrics.get("map50", "—")
    map50_95 = metrics.get("map50_95", "—")
    precision = metrics.get("precision", "—")
    recall = metrics.get("recall", "—")
    suggestion = str(metrics.get("suggestion") or "").strip()
    ai_advice = str(metrics.get("ai_advice") or "").strip()
    ai_source = str(metrics.get("ai_advice_source") or "").strip()

    if ai_source == "llm":
        source_label = "大模型"
    elif ai_source == "fallback":
        source_label = "本地规则"
    elif ai_source:
        source_label = ai_source
    else:
        source_label = ""

    # AI 建议优先；无 AI 时回退演示 suggestion
    advice_block = ""
    if ai_advice:
        advice_block = f"""
<h2>AI 优化建议</h2>
<p style="color:#555;margin:0.4rem 0 0.8rem;">来源：{html.escape(source_label or "AI")}</p>
<pre style="white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:16px;border-radius:8px;line-height:1.65;font-family:inherit;">{html.escape(ai_advice)}</pre>
"""
    elif suggestion:
        advice_block = f"""
<h2>优化建议</h2>
<p>{html.escape(suggestion)}</p>
"""

    now = _beijing_now_str()
    report_html.write_text(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>评估报告 - {html.escape(task_name)}</title>
</head>
<body style="font-family:system-ui,sans-serif;padding:24px;max-width:860px;margin:0 auto;color:#222;">
<h1>检测评估报告</h1>
<p>任务：{html.escape(task_name)}</p>
<ul>
<li>mAP@0.5：{html.escape(str(map50))}</li>
<li>mAP@0.5:0.95：{html.escape(str(map50_95))}</li>
<li>Precision：{html.escape(str(precision))}</li>
<li>Recall：{html.escape(str(recall))}</li>
</ul>
{advice_block}
<p style="color:#888;font-size:0.9rem;">生成时间（北京时间）：{html.escape(now)}</p>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_json, report_html
