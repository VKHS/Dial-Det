from pathlib import Path

from dial_det.examples import synthetic_encoded_stream
from dial_det.io.npz import load_encoded_stream, save_encoded_stream


def test_npz_roundtrip(tmp_path: Path):
    source=synthetic_encoded_stream()
    path=tmp_path/'stream.npz'
    save_encoded_stream(path,source)
    loaded=load_encoded_stream(path)
    assert source.sha256==loaded.sha256
