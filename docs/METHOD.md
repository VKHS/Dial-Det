# Method: from identity evidence to a budgeted report

This guide explains the mathematical and computational contract of DIAL-Det. The formulation follows Sections 1–2 of *DIAL-Det: Learning Identity Mass for Budgeted Detection Allocation* and the detailed definitions in Appendices B–E of its supplement. [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json) identifies the scientific documents. The [implementation notes](../IMPLEMENTATION_NOTES.md) record design details, and the [status map](IMPLEMENTATION_STATUS.md) links each component to code and tests.

## 1. The decision and its inputs

An image supplies a frozen candidate set $V$, with $N\le500$. Each candidate retains its source ID, box, class, native detector score and appearance feature. The decoder chooses $S\subseteq V$ with $|S|\le K$. The membership track preserves the original boxes and scores; learned quality changes which candidates are reported.

Let $H$ be the maximum class-consistent matching size between the complete pool and nonignored ground-truth objects at the audit IoU. Let $C$ be the matching size for the selected report, and $Q=\min(K,H)$. Candidate absence, unavoidable capacity loss and allocation shortfall are respectively $|G|-H$, $H-Q$ and $Q-C$. A decoder acting on a fixed pool directly changes the third term. The benchmark evaluator's assignment and ranking remain a separate measurement.

Ground truth constructs supervised targets and retrospective audits. It does not construct inference support, identity affinities or selected reports. The evidence model receives no budget input; its output can be reused for all requested budgets.

## 2. Learn evidence aligned with the decision

| Component | Implemented contract |
|---|---|
| Overlap support | Same-class pairs at IoU $\ge0.25$ |
| Appearance rescue support | Top-four cosine neighbors inside center distance divided by mean box diagonal $\le2$; symmetric union and self-edges |
| Encoder | Appearance projected to 128 channels; geometry, native score and 16-channel class embedding; two residual graph-attention layers with four heads per branch and a learned gate |
| Quality | Ten nested probabilities for IoU $0.50,0.55,\ldots,0.95$; COCO uses their mean, IoU-0.50 tasks use the first |
| Identity | Symmetric pair head, $r_{ii}=1$ and $r_{ij}=0$ outside support; positive temperature fitted on development data |

The training assignment selects the highest-IoU admissible same-class object, breaking ties by annotation ID. Background candidates have private identities. Ignore handling masks supervision when an identity target cannot be determined safely.

The loss is

$$
\mathcal L=\mathcal L_q+\mathcal L_{\mathrm{pair}}
+0.1\mathcal L_{\mathrm{mass}}+0.05\mathcal L_{\mathrm{trans}},\qquad
\mathcal L_{\mathrm{mass}}=\frac1{|\mathcal U|}\sum_{u\in\mathcal U}(\log d_u-\log d_u^\star)^2.
$$

Here $\mathcal U$ contains the eligible demand nodes: a node's own assignment and every supported neighbor's assignment must be known. The mass loss is zero when $\mathcal U$ is empty; when all candidates are eligible, its normalization reduces to $1/N$. The target $d_u^\star$ counts same-assignment candidates within the complete support neighborhood. Eligibility masks the supervised demand nodes, while each $d_u$ still uses the whole neighborhood.

Transitivity penalizes $[r_{ij}+r_{jk}-1-r_{ik}]_+^2$ on supported wedges, including open wedges where $r_{ik}=0$. Rescue edges address missing same-object links; transitivity discourages a candidate from strongly joining otherwise unrelated objects. These are mechanisms to test with learned evidence, not guarantees that such errors disappear.

Training uses AdamW with learning rate $3\times10^{-4}$, weight decay $10^{-4}$ and 30 default epochs. Checkpoint selection and identity temperature use reserved development data. The frozen detector is not trained by this pipeline. Same-class rescue, hidden width 160, full-edge pair BCE, one-image optimizer steps and exact feature-pooling choices are disclosed [reference choices](IMPLEMENTATION_STATUS.md#explicit-reference-choices).

## 3. Normalize coverage before allocating slots

For the calibrated affinities, compute

$$
s_{ij}=r_{ij}^p,\qquad d_u=\sum_j s_{ju},\qquad
\mu_u=1/d_u,\qquad \widetilde r_{ij}=s_{ij}\mathbf1\{s_{ij}\ge\theta\}.
$$

The mass is **unthresholded**, so even supported affinities removed from the retained graph contribute to $d_u$. Self-affinity ensures $d_u\ge1$. Mass remains fixed through graph splitting, candidate insertions and exchanges. Thresholding raw $r$ instead of $r^p$, or recomputing mass after a split, changes the method.

With empty maxima defined as zero, the objective is

$$
F(S)=\underbrace{\sum_{i\in S}(q_i-\rho)}_{\text{quality and report price}}
+\underbrace{\eta\sum_u\mu_u\max_{i\in S}(q_i\widetilde r_{iu})}_{\text{normalized coverage}}
-\underbrace{\lambda\sum_{\{i,j\}\subseteq S}\widetilde r_{ij}\min(q_i,q_j)}_{\text{redundancy cost}},\quad |S|\le K.
$$

The default tuple is $(\rho,\eta,\lambda,p,\theta)=(0.08,0.2,1.25,2,0.05)$. The redundancy term is a soft cost: distinct objects with overlapping boxes are never excluded by a hard conflict constraint. The objective is nonmonotone submodular; negative marginals and the empty report are valid, so an exactly-$K$ constraint would change the problem.

For an ideal identity block $B$, affinities are one within the block and zero across blocks. Then $d_u=|B|$ and $\sum_{u\in B}\mu_u=1$, irrespective of duplicate count. With $\rho>0$ and $\lambda\ge1$, an optimum contains at most one representative per block. A duplicate-free report has $F(S)=(1+\eta)\sum_{i\in S}q_i-\rho|S|$; when quality has the stated probability interpretation, this equals $(1+\eta)\mathbb E[C(S)]-\rho|S|$ (paper Proposition 1). With learned evidence, fragmented support, false identity bridges and mass error can change the preferred report.

Writing $m_u(S)=\max_{i\in S}q_i\widetilde r_{iu}$, the marginal is

$$
D_j(S)=q_j-\rho+\eta\sum_u\mu_u[q_j\widetilde r_{ju}-m_u(S)]_+
-\lambda\sum_{i\in S}\widetilde r_{ij}\min(q_i,q_j).
$$

The exchange gain is $D_j(S\setminus\{i\})-D_i(S\setminus\{i\})$. It explicitly depends on the alternative candidate $j$, which explains why a fixed pairwise admission threshold cannot in general reproduce the full-report exchange. Closed-form isolated-pair and three-candidate cases are implemented and checked against the full objective; they do not replace general allocation.

## 4. Search and independently check its gap

1. Round objective coefficients to integer units of $\delta=2^{-30}$ and find retained-affinity components. Split components above 256 candidates, recording the removed interactions.
2. Enumerate components of size at most 12. Search larger components with a width-64 insertion beam and deterministic improving single swaps, retaining an incumbent for every feasible cardinality.
3. Form upper curves using coverage-assignment LPs, McCormick product constraints and Boolean-quadric clique cuts. Treat the LP solver's multipliers as proposed witnesses.
4. Independently check residual dual bounds with exact dyadic/rational arithmetic. Combine component lower and upper curves using integer max-plus dynamic programming over sizes $0,\ldots,K$.
5. Add the upper correction for removed coverage and reevaluate the selected report on the original unsplit objective. Replay the certificate from its exported instance and witnesses.

For checked upper bound $B_\delta$ and incumbent $L_\delta$, in objective units,

$$
F^\star-F(S)\le g_F=B_\delta-L_\delta+2\epsilon_g,\qquad
\epsilon_g=\frac\delta2\left(K+N+\binom K2\right).
$$

The implementation caps $K$ at $N$ internally, preserving feasible reports and tightening this envelope. It reports `rounded_optimum` when integer bounds agree, `certified` when $g_F\le0.015\max(B_\delta+\epsilon_g,1)$, and `uncertified` when the checked gap is larger. An uncertified incumbent remains a valid returned report. Numerical-integrity failure returns the declared local-rule `fallback`, with no objective-gap certificate. Its default threshold 0.5 must be tuned on development data for a scientific comparison.

This certificate concerns optimization of the recorded objective and the coefficient-rounding envelope. It does not certify AP, identity accuracy, the data-generating history or population risk. The exact-arithmetic checker and deterministic component splitting are documented implementation choices; [IMPLEMENTATION_NOTES.md](../IMPLEMENTATION_NOTES.md#5-search-and-certificates) supplies their details.

## 5. Measure performance and risk on the correct populations

Matching audits report image-wise waste $(|S|-C)/K$ and shortfall $(Q-C)/Q$, with shortfall zero when $Q=0$. The pooled shortfall $\sum(Q-C)/\sum Q$ has different weighting from the image mean. The retrospective matching-miss bound uses ground-truth reference identities and must remain separate from inference.

The implemented finite-family risk calculation evaluates all 24 fixed $(\rho,\lambda)$ policies, including uncertified and fallback outputs, and uses 48 simultaneous empirical-Bernstein hypotheses for the two risks. A population claim additionally requires the stated independent calibration/deployment setting, a fixed policy family and appropriate detector history. Hashes record identity and consistency; they do not establish these statistical assumptions. Public-checkpoint cross-fitting is ineligible for the focal risk claim.

The [COCO study](COCO_STUDY.md) merges the two disjoint held-out folds before official COCO evaluation for each seed. Paired statistics resample images and encoder seeds together across methods, recompute dataset-level AP for each sampled seed and average across sampled seeds. Repeated image occurrences receive fresh evaluation IDs so bootstrap multiplicity is preserved. The default is 2,000 replicates; the primary interval is the observed contrast plus/minus 1.96 bootstrap standard errors, with a percentile interval also recorded. Full schemas and supported endpoints appear in the [experiment guide](EXPERIMENTS.md#paired-statistics-specification).
