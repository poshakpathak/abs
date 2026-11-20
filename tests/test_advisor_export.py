"""The web advisor's data file is generated from the same tables the API and stream use."""
import json
from pathlib import Path

from abslive import export_advisor, policy

ROOT = Path(__file__).resolve().parents[1]


def test_paper_tables_reproduce_the_published_advisor_file():
    tables = policy.Tables.load(ROOT / "model", version="research")
    meta = json.loads((ROOT / "model" / "research_meta.json").read_text())
    mine = export_advisor.render(export_advisor.advisor_payload(tables, meta))
    assert mine == (ROOT / "web" / "advisor_data.js").read_text()
