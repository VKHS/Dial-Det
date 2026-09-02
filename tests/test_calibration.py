import math
import numpy as np

from dial_det.method.calibration import (
    calibrate_complete_rules,
    empirical_bernstein_ucb,
    simultaneous_log_term,
)


def test_empirical_bernstein_matches_expression():
    losses=np.array([0,.1,.2,.1,0],float)
    log_term=simultaneous_log_term(number_of_rules=2,number_of_risks=2,alpha=.05)
    mean,var,raw,clipped=empirical_bernstein_ucb(losses,logarithmic_term=log_term)
    expected=mean+math.sqrt(2*var*log_term/len(losses))+7*log_term/(3*(len(losses)-1))
    assert np.isclose(raw,expected)
    assert clipped==min(1,max(0,raw))


def test_complete_rule_calibration_family():
    losses={
        'a':{'duplicate':np.zeros(200),'coverage':np.zeros(200)},
        'b':{'duplicate':np.ones(200)*.2,'coverage':np.ones(200)*.2},
    }
    result=calibrate_complete_rules(
        losses,risk_targets={'duplicate':.2,'coverage':.2},alpha=.05,
        utilities={'a':1.0,'b':2.0},calibration_split_id='cal',utility_split_id='util',
    )
    assert result.number_of_rules==2
    assert result.selected_rule_id=='a'
