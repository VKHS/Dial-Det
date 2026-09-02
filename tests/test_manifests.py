from copy import deepcopy
from pathlib import Path
import json

import pytest

from dial_det.config.rule import CompleteRule, load_complete_rule
from dial_det.config.splits import SplitManifest, load_split_manifest

ROOT=Path(__file__).resolve().parents[1]


def test_complete_rule_rejects_placeholder_and_empty_section(tmp_path):
    payload=json.loads((ROOT/'configs/smoke/dial_out_exact.json').read_text())
    CompleteRule(**payload).validate()
    bad=deepcopy(payload); bad['stream']['detector_checkpoint_sha256']='<checkpoint>'
    with pytest.raises(ValueError): CompleteRule(**bad).validate()
    bad=deepcopy(payload); bad['graph']={}
    with pytest.raises(ValueError): CompleteRule(**bad).validate()


def test_split_manifest_checks_actual_ids():
    good=SplitManifest('d','v',('a',),('b',),('c',),('d',),'fixed')
    good.validate()
    bad=SplitManifest('d','v',('a',),('b',),('a',),('d',),'fixed')
    with pytest.raises(ValueError): bad.validate()


def test_smoke_split_hash_matches_rules():
    split=load_split_manifest(ROOT/'configs/smoke/split_manifest.json')
    for name in ('dial_out_at_most.json','dial_out_exact.json','dial_prop_exact.json','dial_prop_cost.json'):
        rule=load_complete_rule(ROOT/'configs/smoke'/name)
        assert rule.statistics['split_manifest_sha256']==split.sha256
