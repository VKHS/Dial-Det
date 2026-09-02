import numpy as np

from dial_det.experiment.statistics import holm_adjustment, paired_tost


def test_holm_is_monotone_in_sorted_order():
    p=np.array([.01,.03,.2])
    adjusted=holm_adjustment(p)
    assert np.all(adjusted>=p)
    assert np.all(adjusted<=1)


def test_tost_equivalence():
    result=paired_tost(np.array([.00,.01,-.01,.02]),lower_margin=-.10,upper_margin=.10)
    assert result.equivalent
