from collections import defaultdict
from pathlib import Path
from typing import Any


def write_samples_markdown(
    rows: list[dict[str, Any]], path: str | Path, *, per_reason: int = 2
) -> None:
    """Write a Markdown digest of up to ``per_reason`` example rows per outcome reason.

    Groups predictions by ``reason`` and renders each sample's prompt and
    completion (truncated), so failures can be eyeballed without opening the
    full JSONL.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        if len(grouped[reason]) < per_reason:
            grouped[reason].append(row)

    lines = ["# Whetstone Samples", ""]
    for reason in sorted(grouped):
        lines.extend([f"## {reason}", ""])
        for row in grouped[reason]:
            lines.extend(
                [
                    f"### {row.get('uid')}",
                    "",
                    f"- passed: `{row.get('passed')}`",
                    f"- reward: `{row.get('reward')}`",
                    f"- template_id: `{row.get('template_id')}`",
                    "",
                    "Prompt:",
                    "",
                    "```text",
                    str(row.get("prompt") or "")[:2000],
                    "```",
                    "",
                    "Completion:",
                    "",
                    "```text",
                    str(row.get("completion") or "")[:2000],
                    "```",
                    "",
                ]
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
