from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_no_article_source_or_pdf_is_distributed():
    forbidden={'.tex','.pdf','.aux','.bbl','.blg','.bib','.synctex'}
    found=[path for path in ROOT.rglob('*') if path.is_file() and path.suffix.lower() in forbidden]
    assert found==[]


def test_no_artificial_gpu_workload_language_or_code():
    terms=('avoid gpu idle','utilization pulse','background ' + 'gemm','idle kills')
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix.lower() == '.py':
            if path.name in {'test_repository_scope.py','release_check.py'}:
                continue
            text=path.read_text(encoding='utf-8',errors='ignore').lower()
            assert all(term not in text for term in terms), path
