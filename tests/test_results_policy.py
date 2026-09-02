import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_only_obtained_article_values_are_in_reported_ledger():
    with (ROOT/'results/reported/paper_results.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==318
    assert all(row['provenance_status']=='reported_in_article' for row in rows)
    text=' '.join(' '.join(row.values()) for row in rows).lower()
    assert 'projected' not in text
    assert 'placeholder' not in text
    assert 'estimated' not in text
    assert 'tbd' not in text
