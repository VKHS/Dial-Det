import numpy as np

from dial_det.method.risks import (
    coverage_shortfall_loss,
    duplicate_multiplicity_loss,
    maximum_iou_assignment,
    proposal_miss_loss,
)


def test_many_to_one_matching_preserves_duplicates():
    predictions = np.array([[0,0,10,10],[1,1,9,9],[20,20,30,30]],float)
    targets = np.array([[0,0,10,10],[20,20,30,30]],float)
    assignment = maximum_iou_assignment(predictions, targets, iou_threshold=.5)
    assert assignment.tolist() == [0,0,1]
    assert np.isclose(duplicate_multiplicity_loss(assignment,budget_k=3),1/3)
    assert coverage_shortfall_loss(assignment,budget_k=3,ground_truth_count=2)==0


def test_proposal_miss_is_separate():
    proposals=np.array([[0,0,10,10]],float)
    targets=np.array([[0,0,10,10],[20,20,30,30]],float)
    assert proposal_miss_loss(proposals,targets,iou_threshold=.5)==.5
