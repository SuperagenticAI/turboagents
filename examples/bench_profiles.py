from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turboagents.bench.kv import build_kv_report
from turboagents.bench.paper import build_paper_report
from turboagents.bench.rag import build_rag_report


print(build_kv_report().to_text())
print()
print(build_rag_report().to_markdown())
print()
print(build_paper_report().to_text())
