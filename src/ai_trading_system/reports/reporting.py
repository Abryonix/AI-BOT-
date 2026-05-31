from __future__ import annotations

from pathlib import Path

import pandas as pd


def model_ranking_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    sort_cols = [c for c in ("composite_score", "sharpe", "calmar") if c in table]
    return table.sort_values(sort_cols, ascending=False).reset_index(drop=True) if sort_cols else table


def write_markdown_report(path: str | Path, title: str, metrics: dict[str, float], ranking: pd.DataFrame) -> None:
    lines = [f"# {title}", "", "## Metrics"]
    lines += [f"- **{key}**: {value:.4f}" for key, value in metrics.items()]
    lines += ["", "## Model Ranking", "", ranking.to_markdown(index=False)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
