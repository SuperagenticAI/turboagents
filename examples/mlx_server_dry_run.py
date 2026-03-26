from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turboagents.cli.serve import run


print(
    run(
        backend="mlx",
        model="mlx-community/Qwen3-0.6B-4bit",
        dry_run=True,
    )
)
