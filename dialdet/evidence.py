"""Learned evidence for DIAL-Det (paper Section 2, Appendix B).

The detector, candidate boxes and native scores are not parameters of this model.
The report budget is deliberately absent from its interface. Sparse messages are
computed only on the two support branches; NxN affinity storage is bounded by
the candidate cap (500).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EvidenceConfig:
    feature_dim: int
    num_classes: int
    appearance_dim: int = 128
    class_embedding_dim: int = 16
    hidden_dim: int = 160
    layers: int = 2
    heads: int = 4
    overlap_threshold: float = 0.25
    rescue_k: int = 4
    distance_gate: float = 2.0
    rescue_same_class: bool = True
    use_appearance: bool = True
    use_rescue: bool = True


def pair_geometry(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return directed geometry, symmetric geometry, and IoU matrices.

    Geometry features are dx/mean-width, dy/mean-height, log(wj/wi),
    log(hj/hi), IoU and normalized centre distance. Absolute values of the first
    four entries make the identity-head geometry exactly symmetric.
    """
    boxes = boxes.float()
    size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    centre = (boxes[:, 2:] + boxes[:, :2]) * 0.5
    mean_size = (size[:, None] + size[None, :]) * 0.5
    relative_centre = (centre[None, :] - centre[:, None]) / mean_size
    relative_size = size[None, :].log() - size[:, None].log()
    lo = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    hi = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    area = size.prod(-1)
    iou = intersection / (area[:, None] + area[None, :] - intersection).clamp_min(1e-12)
    diagonal = size.norm(dim=-1)
    distance = (centre[:, None] - centre[None, :]).norm(dim=-1)
    distance = distance / ((diagonal[:, None] + diagonal[None, :]) * 0.5).clamp_min(1e-6)
    direction = torch.cat((relative_centre, relative_size, iou[..., None], distance[..., None]), dim=-1)
    symmetry = torch.cat((direction[..., :4].abs(), direction[..., 4:]), dim=-1)
    return direction, symmetry, iou


@torch.no_grad()
def build_support(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    appearance: torch.Tensor,
    overlap_threshold: float = 0.25,
    rescue_k: int = 4,
    distance_gate: float = 2.0,
    rescue_same_class: bool = True,
    use_rescue: bool = True,
) -> dict[str, torch.Tensor]:
    """Construct symmetrized overlap and appearance-rescue supports, with self.

    Rescue ties are resolved by stable input order (export order is score then
    source ID); candidates never appear as their own top-k rescue neighbour.
    Self-edges are added afterwards. The original, pre-message appearance is
    used, so edge identities cannot drift with learned graph embeddings.
    """
    n = boxes.shape[0]
    direction, symmetry, iou = pair_geometry(boxes)
    same_class = classes[:, None] == classes[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=boxes.device)
    overlap = ((iou >= overlap_threshold) & same_class) | eye
    rescue = torch.zeros_like(overlap)
    if use_rescue and rescue_k > 0 and n > 1:
        feature = F.normalize(appearance.float(), p=2, dim=-1, eps=1e-12)
        similarity = feature @ feature.T
        eligible = (direction[..., -1] <= distance_gate) & ~eye
        if rescue_same_class:
            eligible &= same_class
        similarity = similarity.masked_fill(~eligible, -torch.inf)
        indices = torch.argsort(similarity, dim=1, descending=True, stable=True)[:, :min(rescue_k, n - 1)]
        rows = torch.arange(n, device=boxes.device)[:, None].expand_as(indices)
        rescue[rows, indices] = eligible[rows, indices]
        rescue = rescue | rescue.T
    rescue = rescue | eye
    return {"overlap": overlap, "rescue": rescue, "support": overlap | rescue,
            "geometry": direction, "symmetric_geometry": symmetry, "iou": iou}


class SparseBranchAttention(nn.Module):
    """Equation B.2, with additive linear maps equivalent to W[hi|hj|psi]."""
    def __init__(self, hidden_dim: int, heads: int, geometry_dim: int = 6):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.left = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.right = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(geometry_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, h: torch.Tensor, support: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        if n == 0:
            return h
        destination, source = support.nonzero(as_tuple=True)
        latent = self.left(h)[destination] + self.right(h)[source] + self.edge(geometry[destination, source].to(h.dtype))
        latent = F.leaky_relu(latent.reshape(-1, self.heads, self.head_dim), negative_slope=0.2)
        logits = (latent * self.attention).sum(-1).float()
        # Stable segment softmax: one segment per destination, per head.
        index = destination[:, None].expand(-1, self.heads)
        maxima = torch.full((n, self.heads), -torch.inf, device=h.device)
        maxima.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
        exponential = (logits - maxima[destination]).exp()
        denominator = torch.zeros((n, self.heads), device=h.device)
        denominator.scatter_add_(0, index, exponential)
        alpha = exponential / denominator[destination].clamp_min(1e-12)
        values = self.value(h).reshape(n, self.heads, self.head_dim)
        messages = alpha[..., None].to(values.dtype) * values[source]
        aggregate = torch.zeros_like(values)
        aggregate.index_add_(0, destination, messages)
        return aggregate.reshape(n, -1)


class TwoBranchLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.overlap_attention = SparseBranchAttention(hidden_dim, heads)
        self.rescue_attention = SparseBranchAttention(hidden_dim, heads)
        self.gate = nn.Linear(3 * hidden_dim + 2, 1)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, graph: dict[str, torch.Tensor], use_rescue: bool = True) -> torch.Tensor:
        mo = self.overlap_attention(h, graph["overlap"], graph["geometry"])
        if not use_rescue:
            return h + self.output(F.gelu(self.normalization(mo)))
        mr = self.rescue_attention(h, graph["rescue"], graph["geometry"])
        degree_o = graph["overlap"].sum(-1, keepdim=True).to(h.dtype).log1p()
        degree_r = graph["rescue"].sum(-1, keepdim=True).to(h.dtype).log1p()
        gate = torch.sigmoid(self.gate(torch.cat((h, mo, mr, degree_o, degree_r), -1)))
        combined = gate * mo + (1.0 - gate) * mr
        return h + self.output(F.gelu(self.normalization(combined)))


class EvidenceModel(nn.Module):
    """Two-branch GAT with nested quality and symmetric identity heads."""
    def __init__(self, feature_dim: int, num_classes: int, **kwargs: Any):
        super().__init__()
        self.config = EvidenceConfig(feature_dim, num_classes, **kwargs)
        c = self.config
        if c.feature_dim < 1 or c.num_classes < 1 or c.layers < 1:
            raise ValueError("feature_dim, num_classes and layers must be positive")
        self.appearance_projection = nn.Linear(c.feature_dim, c.appearance_dim)
        self.class_embedding = nn.Embedding(c.num_classes, c.class_embedding_dim)
        self.input_projection = nn.Linear(c.appearance_dim + 4 + 1 + c.class_embedding_dim, c.hidden_dim)
        self.layers = nn.ModuleList(TwoBranchLayer(c.hidden_dim, c.heads) for _ in range(c.layers))
        self.quality_head = nn.Sequential(nn.Linear(c.hidden_dim, c.hidden_dim), nn.GELU(), nn.Linear(c.hidden_dim, 10))
        self.identity_head = nn.Sequential(nn.Linear(2 * c.hidden_dim + 6, c.hidden_dim), nn.GELU(), nn.Linear(c.hidden_dim, 1))
        # Small initial positive decrements avoid collapsing all high-IoU heads.
        with torch.no_grad():
            self.quality_head[-1].bias[1:] = -3.0

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def forward(self, boxes: torch.Tensor, classes: torch.Tensor, scores: torch.Tensor,
                features: torch.Tensor, image_size: tuple[int, int], temperature: float = 1.0,
                p: float = 2.0, quality_mode: str = "coco") -> dict[str, torch.Tensor]:
        if quality_mode not in {"coco", "iou50"}:
            raise ValueError("quality_mode must be 'coco' or 'iou50'")
        if not temperature > 0 or not p >= 1:
            raise ValueError("temperature must be positive and p >= 1")
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("appearance feature dimension differs from model")
        n = boxes.shape[0]
        if n > 500:
            raise ValueError("paper candidate cap exceeded: N must be <= 500")
        if classes.numel() and (classes.min() < 0 or classes.max() >= self.config.num_classes):
            raise ValueError("class index lies outside the embedding table")
        h_image, w_image = image_size
        if h_image <= 0 or w_image <= 0:
            raise ValueError("image_size must contain positive height and width")
        c = self.config
        boxes = boxes.float()
        features = features.float()
        # In the no-appearance ablation, both message features and rescue ranking
        # lose visual information; stable geometry-gated ties remain explicit.
        appearance = features if c.use_appearance else torch.zeros_like(features)
        graph = build_support(boxes, classes, appearance, c.overlap_threshold, c.rescue_k,
                              c.distance_gate, c.rescue_same_class, c.use_rescue)
        normalizer = boxes.new_tensor([w_image, h_image, w_image, h_image])
        geometry = boxes / normalizer
        app = F.gelu(self.appearance_projection(appearance))
        h = self.input_projection(torch.cat((app, geometry, scores.float()[:, None], self.class_embedding(classes)), -1))
        for layer in self.layers:
            h = layer(h, graph, use_rescue=c.use_rescue)
        raw_quality = self.quality_head(h).float()
        quality_logits = torch.cat((raw_quality[:, :1], raw_quality[:, :1] - F.softplus(raw_quality[:, 1:]).cumsum(-1)), -1)
        quality_probs = quality_logits.sigmoid()
        q = quality_probs.mean(-1) if quality_mode == "coco" else quality_probs[:, 0]
        pair_mask = graph["support"] & torch.triu(torch.ones_like(graph["support"]), diagonal=1)
        left, right = pair_mask.nonzero(as_tuple=True)
        sym = graph["symmetric_geometry"][left, right].to(h.dtype)
        pair_features = torch.cat(((h[left] - h[right]).abs(), h[left] * h[right], sym), -1)
        pair_values = self.identity_head(pair_features).flatten().float()
        logits = h.new_zeros((n, n), dtype=torch.float32)
        logits[left, right] = pair_values
        logits[right, left] = pair_values
        r = h.new_zeros((n, n), dtype=torch.float32)
        probabilities = (pair_values / temperature).sigmoid()
        r[left, right] = probabilities
        r[right, left] = probabilities
        r = r + torch.eye(n, device=h.device, dtype=r.dtype)
        d = r.pow(p).sum(dim=0)
        return {"q": q, "r": r, "d": d, "quality_logits": quality_logits,
                "quality_probs": quality_probs, "pair_logits": logits,
                "pair_mask": pair_mask, "embeddings": h, **graph}


def transitivity_loss(r: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, int]:
    """All length-two support wedges, including absent closing edges.

    Endpoints are unordered (the reverse orientation has the identical value).
    Centres are distinct from both endpoints. No random triangle subsampling is
    used, and no dense N-by-N-by-N tensor is allocated.
    """
    total = r.sum() * 0.0
    count = 0
    n = r.shape[0]
    for centre in range(n):
        neighbours = support[centre].nonzero(as_tuple=True)[0]
        neighbours = neighbours[neighbours != centre]
        degree = neighbours.numel()
        if degree < 2:
            continue
        pairs = torch.triu_indices(degree, degree, 1, device=r.device)
        a, b = neighbours[pairs[0]], neighbours[pairs[1]]
        violations = (r[a, centre] + r[centre, b] - 1.0 - r[a, b]).clamp_min(0)
        total = total + violations.square().sum()
        count += violations.numel()
    return total / max(count, 1), count


def evidence_losses(output: dict[str, torch.Tensor], assignments: torch.Tensor,
                    quality_targets: torch.Tensor, valid_candidates: torch.Tensor | None = None,
                    beta: float = 0.1, gamma: float = 0.05, p: float = 2.0) -> dict[str, torch.Tensor | int]:
    """Equation B.4, using all natural support pairs (P=Q, weight one).

    Unknown/ignored identities cannot give a target for their neighbours' full
    masses. Mass supervision is therefore masked for every node adjacent to an
    ignored candidate; inference masses still include every support neighbour.
    This avoids silently changing the normalization to a sampled/masked mass.
    """
    r, support = output["r"], output["support"]
    valid = torch.ones_like(assignments, dtype=torch.bool) if valid_candidates is None else valid_candidates.bool()
    zero = output["quality_logits"].sum() * 0.0 + r.sum() * 0.0
    lq = F.binary_cross_entropy_with_logits(output["quality_logits"][valid], quality_targets[valid].float()) if valid.any() else zero
    equal_identity = assignments[:, None] == assignments[None, :]
    pair_mask = output["pair_mask"] & valid[:, None] & valid[None, :]
    lpair = F.binary_cross_entropy_with_logits(output["pair_logits"][pair_mask], equal_identity[pair_mask].float()) if pair_mask.any() else zero
    target_mass = (support & equal_identity).sum(dim=0).to(r.dtype).clamp_min(1.0)
    full_mass = r.pow(p).sum(dim=0).clamp_min(1.0)
    mass_valid = valid & ~(support & ~valid[:, None]).any(dim=0)
    lm = (full_mass[mass_valid].log() - target_mass[mass_valid].log()).square().mean() if mass_valid.any() else zero
    if gamma:
        lt, wedges = transitivity_loss(r, support)
    else:
        lt, wedges = zero, 0
    total = lq + lpair + beta * lm + gamma * lt
    return {"total": total, "quality": lq, "pair": lpair, "mass": lm, "transitivity": lt,
            "pairs": int(pair_mask.sum().item()), "wedges": wedges, "mass_nodes": int(mass_valid.sum().item())}
