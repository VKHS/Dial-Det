"""Framework-neutral construction of a validated encoded candidate stream."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dial_det.data.stream import EncodedCandidateStream


def build_encoded_stream(
    *,
    candidate_ids: Sequence[str],
    boxes_xyxy: np.ndarray,
    detector_scores: np.ndarray,
    class_probabilities: np.ndarray,
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edge_geometry: np.ndarray,
    pressure_features: np.ndarray,
    raw_benefit_out: np.ndarray,
    uncertainty_out: np.ndarray,
    object_probability: np.ndarray,
    localization_quality: np.ndarray,
    coverage_probability: np.ndarray,
    roi_success_probability: np.ndarray,
    directional_identity_logits: np.ndarray,
    compatibility: np.ndarray,
    graph_degree: np.ndarray,
    measured_cost: np.ndarray,
    eligible_identity_pairs: np.ndarray,
    detector_id: str,
    detector_checkpoint_sha256: str,
    evidence_checkpoint_sha256: str,
    candidate_interface_id: str,
    candidate_order: str,
    feature_taps: list[str],
    preprocessing: dict[str, Any],
    class_mapping: dict[str, str],
    relation_types: list[str],
    active_relations: list[str],
    export_software_version: str,
) -> EncodedCandidateStream:
    stream = EncodedCandidateStream(
        candidate_ids=np.asarray(candidate_ids).astype(str),
        boxes_xyxy=boxes_xyxy,
        detector_scores=detector_scores,
        class_probabilities=class_probabilities,
        node_features=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_geometry=edge_geometry,
        pressure_features=pressure_features,
        raw_benefit_out=raw_benefit_out,
        uncertainty_out=uncertainty_out,
        object_probability=object_probability,
        localization_quality=localization_quality,
        coverage_probability=coverage_probability,
        roi_success_probability=roi_success_probability,
        directional_identity_logits=directional_identity_logits,
        compatibility=compatibility,
        graph_degree=graph_degree,
        measured_cost=measured_cost,
        eligible_identity_pairs=eligible_identity_pairs,
        metadata={
            "detector_id": detector_id,
            "detector_checkpoint_sha256": detector_checkpoint_sha256,
            "evidence_checkpoint_sha256": evidence_checkpoint_sha256,
            "candidate_interface_id": candidate_interface_id,
            "candidate_order": candidate_order,
            "feature_taps": feature_taps,
            "preprocessing": preprocessing,
            "class_mapping": class_mapping,
            "relation_types": relation_types,
            "active_relations": active_relations,
            "export_software_version": export_software_version,
        },
    )
    stream.validate()
    return stream
