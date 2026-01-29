from __future__ import annotations

import shutil
import sys
import textwrap
from typing import Iterable, Tuple

from app.graph.judge_models import JudgeEvaluation, PublicReport


def _clamp_int(v: object, lo: int, hi: int, default: int) -> int:
    try:
        i = int(v)  # type: ignore[arg-type]
    except Exception:
        return default
    return max(lo, min(hi, i))


def _clamp_float(v: object, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
    except Exception:
        return default
    return max(lo, min(hi, f))


def _unicode_ok() -> bool:
    # Heuristic: on Windows, stdout is often cp1252 unless user opted into UTF-8.
    # Avoid crashing the CLI due to UnicodeEncodeError.
    try:
        enc = (sys.stdout.encoding or "").lower()
        return "utf" in enc
    except Exception:
        return False


def render_stars(overall_score_0_to_10: object) -> str:
    overall = _clamp_float(overall_score_0_to_10, 0.0, 10.0, 0.0)
    stars = int(round(overall / 2.0))
    stars = max(0, min(5, stars))
    if _unicode_ok():
        return "★" * stars + "☆" * (5 - stars)
    return "*" * stars + "." * (5 - stars)


def render_bar(score: object, max_score: object = 10, *, width: int = 10) -> str:
    max_s = _clamp_int(max_score, 1, 100, 10)
    sc = _clamp_int(score, 0, max_s, 0)
    # normalize to width cells
    filled = int(round((sc / max_s) * width)) if max_s else 0
    filled = max(0, min(width, filled))
    if _unicode_ok():
        return "█" * filled + "░" * (width - filled)
    return "#" * filled + "-" * (width - filled)


def _truncate(items: Iterable[str], n: int) -> list[str]:
    out: list[str] = []
    for it in items:
        s = (it or "").strip()
        if not s:
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


def wrap_text(text: str, width: int) -> list[str]:
    txt = (text or "").strip()
    if not txt:
        return [""]
    return textwrap.wrap(txt, width=width, break_long_words=False, break_on_hyphens=False) or [txt]


def _term_width(default: int = 60) -> int:
    try:
        w = shutil.get_terminal_size(fallback=(default, 20)).columns
        return max(50, min(100, w))
    except Exception:
        return default


def _box(width: int, lines: list[str]) -> str:
    inner = width - 2

    def pad(line: str) -> str:
        # hard clamp
        if len(line) > inner:
            line = line[:inner]
        return line + " " * (inner - len(line))

    if _unicode_ok():
        top = "┌" + "─" * inner + "┐"
        bot = "└" + "─" * inner + "┘"
        body = ["│" + pad(l) + "│" for l in lines]
        return "\n".join([top] + body + [bot])

    top = "+" + "-" * inner + "+"
    bot = "+" + "-" * inner + "+"
    body = ["|" + pad(l) + "|" for l in lines]
    return "\n".join([top] + body + [bot])


def _status_line(recommendation: str | None) -> Tuple[str, str]:
    rec = (recommendation or "").strip().lower()
    if rec == "publish":
        return ("✅" if _unicode_ok() else "OK"), "APPROVED FOR PUBLICATION"
    if rec == "revise":
        return ("🟡" if _unicode_ok() else "WARN"), "NEEDS REVISION"
    if rec == "reject":
        return ("⛔" if _unicode_ok() else "NO"), "REJECTED (DO NOT PUBLISH)"
    return "?", "UNKNOWN"


def render_evaluation_card(
    *,
    report: PublicReport,
    evaluation: JudgeEvaluation,
    width: int | None = None,
) -> str:
    w = width or _term_width(60)
    inner = w - 2

    title = f'Research Report: "{(report.topic or "").strip()}"'
    title = title[:inner]

    overall = _clamp_float(evaluation.overall_score, 0.0, 10.0, 0.0)
    stars = render_stars(overall)
    overall_line = f"Overall Quality: {stars}  {overall:.1f}/10"

    fa = evaluation.factual_accuracy
    co = evaluation.completeness

    fa_score = _clamp_int(fa.score, 0, 10, 0)
    co_score = _clamp_int(co.score, 0, 10, 0)

    fa_bar = render_bar(fa_score, fa.max_score, width=10)
    co_bar = render_bar(co_score, co.max_score, width=10)

    check = "✓" if _unicode_ok() else "+"
    warn = "⚠" if _unicode_ok() else "!"
    bullet = "•" if _unicode_ok() else "*"

    fa_bullets: list[str] = []
    for s in _truncate(fa.strengths or [], 2):
        fa_bullets.append(f"{check} {s}")
    for s in _truncate(fa.weaknesses or [], 2 - len(fa_bullets)):
        fa_bullets.append(f"{warn} {s}")
    if not fa_bullets:
        fa_bullets.append(f"{bullet} No notable issues detected.")

    co_bullets: list[str] = []
    for s in _truncate(co.strengths or [], 2):
        co_bullets.append(f"{check} {s}")
    weak = _truncate(co.weaknesses or [], 2)
    if weak:
        # Prefer "Missing:" formatting for completeness
        co_bullets.append(f"{warn} Missing: {weak[0]}")
        if len(weak) > 1:
            co_bullets.append(f"{warn} {weak[1]}")

    # Coverage statuses (new): if something is explicitly not applicable, surface it as neutral.
    try:
        na_keys = [k for k, v in (co.coverage or {}).items() if getattr(v, "status", None) == "not_applicable"]
    except Exception:
        na_keys = []
    if na_keys:
        label = "N/A" if _unicode_ok() else "N/A"
        co_bullets.append(f"{bullet} {label}: {', '.join(na_keys[:3])}")
    if not co_bullets:
        co_bullets.append(f"{bullet} No notable issues detected.")

    assessment_lines = wrap_text(evaluation.overall_assessment or "", width=inner - 2)
    quoted = [f'"{assessment_lines[0]}' if assessment_lines else '""']
    if len(assessment_lines) > 1:
        for mid in assessment_lines[1:-1]:
            quoted.append(mid)
        quoted.append(assessment_lines[-1] + '"')
    else:
        quoted[0] = quoted[0] + '"'

    icon, status = _status_line(getattr(evaluation, "recommendation", None))

    lines: list[str] = []
    lines.append(title)
    lines.append("")
    lines.append(overall_line)
    lines.append("")
    lines.append("📊 Detailed Scores:" if _unicode_ok() else "Detailed Scores:")
    if _unicode_ok():
        lines.append(f" ├ Factual Accuracy: {fa_bar}  {fa_score}/10")
        for b in fa_bullets[:2]:
            for wl in wrap_text(b, inner - 4):
                lines.append(f" │  {wl}")
        lines.append(f" └ Completeness:     {co_bar}  {co_score}/10")
    else:
        lines.append(f" - Factual Accuracy: {fa_bar}  {fa_score}/10")
        for b in fa_bullets[:2]:
            for wl in wrap_text(b, inner - 4):
                lines.append(f"   {wl}")
        lines.append(f" - Completeness:     {co_bar}  {co_score}/10")
    for b in co_bullets[:2]:
        for wl in wrap_text(b, inner - 4):
            lines.append(f"    {wl}")
    lines.append("")
    lines.append("💡 Judge's Assessment:" if _unicode_ok() else "Judge's Assessment:")
    for q in quoted:
        for wl in wrap_text(q, inner):
            lines.append(wl)
    lines.append("")
    lines.append(f"{icon} Status: {status}")
    lines.append("")
    lines.append("[Publish Now] [Request Revision]")

    return _box(w, lines)


def render_missing_evaluation_card(*, report: PublicReport, width: int | None = None) -> str:
    w = width or _term_width(60)
    inner = w - 2
    title = f'Research Report: "{(report.topic or "").strip()}"'
    title = title[:inner]

    lines = [
        title,
        "",
        ("⚠ Judge evaluation unavailable." if _unicode_ok() else "! Judge evaluation unavailable."),
        "",
        "[Publish Now] [Request Revision]",
    ]
    return _box(w, lines)
