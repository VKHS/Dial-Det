"""Semantic tests for learned evidence and the frozen export boundary.

Torch-dependent tests skip cleanly in the core-only environment. No trained
detector weights, image datasets or internet access are used by these tests.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
    from dialdet.evidence import EvidenceModel, build_support, evidence_losses, transitivity_loss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "install torch to exercise the evidence model")
class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_rescue_gate_class_symmetry_and_self(self):
        boxes = torch.tensor([[0, 0, 10, 10], [15, 0, 25, 10], [1000, 0, 1010, 10], [0, 0, 10, 10]], dtype=torch.float)
        classes = torch.tensor([0, 0, 0, 1])
        features = torch.tensor([[1., 0], [1., 0], [1., 0], [1., 0]])
        graph = build_support(boxes, classes, features, rescue_k=1)
        self.assertFalse(bool(graph["overlap"][0, 1]))
        self.assertTrue(bool(graph["rescue"][0, 1]))
        self.assertFalse(bool(graph["support"][0, 2]))
        self.assertFalse(bool(graph["support"][0, 3]))
        for branch in ("overlap", "rescue", "support"):
            self.assertTrue(torch.equal(graph[branch], graph[branch].T))
            self.assertTrue(bool(graph[branch].diagonal().all()))

    def test_nested_quality_symmetric_identity_and_full_mass(self):
        torch.manual_seed(7)
        model = EvidenceModel(4, 2, hidden_dim=16)
        output = model(torch.tensor([[0., 0, 10, 10], [1., 1, 11, 11], [60., 60, 65, 65]]),
                       torch.tensor([0, 0, 1]), torch.tensor([.9, .8, .6]), torch.rand(3, 4), (100, 100), p=2)
        self.assertTrue(bool((output["quality_probs"][:, 1:] <= output["quality_probs"][:, :-1]).all()))
        self.assertTrue(torch.equal(output["r"], output["r"].T))
        self.assertTrue(torch.equal(output["r"].diagonal(), torch.ones(3)))
        self.assertTrue(bool((output["r"][~output["support"]] == 0).all()))
        self.assertTrue(torch.allclose(output["d"], output["r"].square().sum(0)))
        self.assertTrue(torch.allclose(output["q"], output["quality_probs"].mean(-1)))
        recalibrated = model(torch.tensor([[0., 0, 10, 10], [1., 1, 11, 11], [60., 60, 65, 65]]),
                             torch.tensor([0, 0, 1]), torch.tensor([.9, .8, .6]), torch.rand(3, 4),
                             (100, 100), temperature=2.0, quality_mode="iou50")
        self.assertTrue(torch.allclose(recalibrated["q"], recalibrated["quality_probs"][:, 0]))

    def test_open_wedge_transitivity_is_penalized(self):
        r = torch.tensor([[1., .9, 0], [.9, 1., .9], [0, .9, 1.]], requires_grad=True)
        support = r.detach() > 0
        loss, count = transitivity_loss(r, support)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(loss), .64, places=6)
        loss.backward()
        self.assertGreater(float(r.grad[0, 1]), 0)

    def test_log_mass_loss_reaches_pair_parameters(self):
        theta = torch.tensor(0.1, requires_grad=True)
        off_diagonal = torch.tensor([[0., 1.], [1., 0.]])
        r = torch.eye(2) + theta.sigmoid() * off_diagonal
        support = torch.ones(2, 2, dtype=torch.bool)
        output = {"r": r, "support": support, "pair_mask": torch.triu(support, 1),
                  "pair_logits": theta.expand(2, 2), "quality_logits": torch.zeros(2, 10, requires_grad=True)}
        loss = evidence_losses(output, torch.tensor([0, 0]), torch.ones(2, 10), p=2, gamma=0)
        loss["mass"].backward()
        self.assertLess(float(theta.grad), 0)
        self.assertEqual(loss["pairs"], 1)
        self.assertEqual(loss["mass_nodes"], 2)
        ignored = evidence_losses(output, torch.tensor([0, 0]), torch.ones(2, 10),
                                  valid_candidates=torch.tensor([True, False]), gamma=0)
        self.assertEqual(ignored["mass_nodes"], 0)

    def test_empty_image_and_checkpoint_resume(self):
        from dialdet.data import CandidateRecord
        from dialdet.training import train, load_evidence_checkpoint, predict_record

        model = EvidenceModel(4, 2, hidden_dim=16)
        empty = model(torch.empty(0, 4), torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0, 4), (100, 100))
        self.assertEqual(tuple(empty["r"].shape), (0, 0))

        def record(image_id, role, seed):
            return CandidateRecord(image_id=image_id, ids=np.arange(3),
                                   boxes=np.array([[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]]),
                                   classes=np.zeros(3, dtype=np.int64), scores=np.array([.9, .8, .7]),
                                   features=np.random.default_rng(seed).normal(size=(3, 4)).astype(np.float32),
                                   image_size=(64, 64), gt_boxes=np.array([[0, 0, 10, 10], [30, 30, 40, 40]]),
                                   gt_classes=np.zeros(2, dtype=np.int64), gt_ids=np.array([10, 20]),
                                   gt_ignore=np.zeros(2, dtype=bool), metadata={"role": role})

        fitting = [record("train1", "train", 1), record("train2", "train", 2)]
        dev = [record("dev1", "dev", 3)]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            options = {"model_config": {"hidden_dim": 16}, "seed": 4, "gamma": .05}
            first = train(fitting, dev, root / "resumed", epochs=1, **options)
            resumed = train(fitting, dev, root / "resumed", epochs=2, resume=first["last_checkpoint"], **options)
            whole = train(fitting, dev, root / "whole", epochs=2, **options)
            state_a = torch.load(resumed["last_checkpoint"], weights_only=True)["model_state"]
            state_b = torch.load(whole["last_checkpoint"], weights_only=True)["model_state"]
            for name in state_a:
                self.assertTrue(torch.equal(state_a[name], state_b[name]), name)
            loaded, meta = load_evidence_checkpoint(resumed["checkpoint"])
            self.assertEqual(meta["train_image_ids"], ["train1", "train2"])
            self.assertEqual(meta["dev_image_ids"], ["dev1"])
            prediction = predict_record(dev[0], loaded, temperature=meta["temperature"])
            self.assertEqual(prediction["q"].shape, (3,))
            self.assertTrue(np.allclose(prediction["d"], (prediction["r"] ** 2).sum(0)))
            self.assertGreater(meta["temperature"], 0)
            self.assertLessEqual(meta["temperature_fit"]["loss_after"], meta["temperature_fit"]["loss_before"])

    def test_audit_images_cannot_train_temperature(self):
        from dialdet.data import CandidateRecord
        from dialdet.training import fit_temperature
        record = CandidateRecord("audit", np.arange(1), np.array([[0, 0, 10, 10]]),
                                 np.zeros(1, dtype=np.int64), np.array([.9]), np.ones((1, 4)), (10, 10),
                                 metadata={"role": "audit"})
        with self.assertRaisesRegex(ValueError, "audit images"):
            fit_temperature(EvidenceModel(4, 1), [record])

    def test_configured_appearance_and_rescue_ablations_change_actual_evidence(self):
        folder = Path(__file__).resolve().parents[1] / "configs" / "ablations"
        boxes = torch.tensor([[0., 0, 10, 10], [12., 0, 22, 10], [24., 0, 34, 10]])
        classes, scores = torch.zeros(3, dtype=torch.long), torch.tensor([.9, .8, .7])
        torch.manual_seed(44)
        features = torch.randn(3, 4)
        no_appearance = json.loads((folder / "no_appearance.json").read_text())["model"]
        model = EvidenceModel(4, 2, **no_appearance).eval()
        first = model(boxes, classes, scores, features, (50, 50))
        second = model(boxes, classes, scores, features.flip(0) * 10., (50, 50))
        for key in ("q", "r", "d", "support"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(bool(first["support"][0, 1]))

        no_rescue = json.loads((folder / "no_rescue.json").read_text())["model"]
        output = EvidenceModel(4, 2, **no_rescue).eval()(boxes, classes, scores, features, (50, 50))
        self.assertTrue(torch.equal(output["support"], output["overlap"]))
        self.assertFalse(bool(output["support"][0, 1]))
        self.assertTrue(bool(output["support"].diagonal().all()))

    def test_configured_loss_ablations_remove_only_the_selected_penalty(self):
        folder = Path(__file__).resolve().parents[1] / "configs" / "ablations"
        r = torch.tensor([[1., .9, 0.], [.9, 1., .9], [0., .9, 1.]])
        support = r > 0
        output = {"r": r, "support": support, "pair_mask": torch.triu(support, 1),
                  "pair_logits": torch.zeros(3, 3), "quality_logits": torch.zeros(3, 10)}
        losses = {}
        for variant in ("full", "no_mass_loss", "no_transitivity"):
            settings = json.loads((folder / (variant + ".json")).read_text())["training"]
            losses[variant] = evidence_losses(output, torch.zeros(3, dtype=torch.long),
                      torch.ones(3, 10), beta=settings["beta"], gamma=settings["gamma"])
        full = losses["full"]
        self.assertGreater(float(full["mass"]), 0)
        self.assertGreater(float(full["transitivity"]), 0)
        self.assertAlmostEqual(float(full["total"] - losses["no_mass_loss"]["total"]),
                               .1 * float(full["mass"]), places=6)
        self.assertAlmostEqual(float(full["total"] - losses["no_transitivity"]["total"]),
                               .05 * float(full["transitivity"]), places=6)

    def test_real_torchvision_export_boundary_without_pretrained_weights(self):
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
        except (ImportError, RuntimeError) as error:
            self.skipTest(f"working torchvision unavailable: {error}")
        from dialdet.exporter import export_image

        torch.manual_seed(3)
        detector = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=3,
                                          min_size=64, max_size=64, rpn_pre_nms_top_n_test=40,
                                          rpn_post_nms_top_n_test=20).eval()
        image = torch.rand(3, 64, 64)
        record = export_image(detector, image, "synthetic-no-pretrained-weights", score_floor=0,
                              max_candidates=20, metadata={"validation_only": True})
        self.assertEqual(len(record.ids), 20)
        self.assertEqual(len(np.unique(record.ids)), 20)
        self.assertEqual(record.features.shape, (20, 256))
        self.assertTrue(np.all(record.scores[:-1] >= record.scores[1:]))
        self.assertTrue(np.all(record.boxes >= 0) and np.all(record.boxes <= 64))
        self.assertFalse(record.metadata["final_detection_nms_applied"])
        self.assertTrue(record.metadata["rpn_nms_preserved"])
        self.assertTrue(np.all(np.isin(record.classes, [1, 2])))


if __name__ == "__main__":
    unittest.main()
