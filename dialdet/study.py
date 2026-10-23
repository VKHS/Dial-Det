"""Sequential, auditable public-checkpoint COCO cross-fit studies.

Planning reads and validates the frozen inputs but never trains a model. Execution
delegates to the same public CLI used for individual experiments. This module is
an experiment runner, not a cluster scheduler or a claim of paper reproduction.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from .artifacts import invocation_inputs, source_fingerprint, verify_run
from .data import (atomic_json, classify_synthetic, load_record, manifest_document,
                   record_hash, sha256_file)

PLAN_SCHEMA = "dialdet-coco-study-plan-v1"
STATUS_SCHEMA = "dialdet-coco-study-status-v1"
_METRICS = {"AP", "AP50", "AP75", "mean_image_shortfall", "mean_image_waste", "pooled_shortfall"}
_METHODS = {"dialdet", "local", "greedy", "nms", "soft_nms"}
_CONTROL_FILES = {"RUN_STATUS.json", "FAILURE.json", "JOB_STATUS.json", "JOB_STATUS.txt", "STUDY_LOCK"}
_MODEL_FIELDS = {"feature_dim", "num_classes", "appearance_dim", "class_embedding_dim", "hidden_dim",
                 "layers", "heads", "overlap_threshold", "rescue_k", "distance_gate", "rescue_same_class",
                 "use_appearance", "use_rescue"}
_TRAINING_FIELDS = {"learning_rate", "weight_decay", "beta", "gamma", "assignment_iou", "gradient_clip", "num_classes"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def _object(value, allowed, required, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) - allowed:
        raise ValueError(f"Unknown {name} fields: {sorted(set(value) - allowed)}")
    if required - set(value):
        raise ValueError(f"Missing {name} fields: {sorted(required - set(value))}")


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'unbounded'}]")
    return value


def _integers(value, name, minimum, minimum_length):
    if not isinstance(value, list) or len(value) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} integers")
    values = [_integer(item, name, minimum, 2**32 - 1) for item in value]
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique integers")
    return values


def _file(value, parent, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must name an existing file")
    path = (parent / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return str(path)


def _normalize(specification):
    path = Path(specification).resolve()
    spec = _read_json(path)
    allowed = {"schema_version", "source_manifest", "coco_annotations", "seeds", "split_seed",
               "dev_fraction", "epochs", "budgets", "device", "config", "comparisons"}
    _object(spec, allowed, {"schema_version", "source_manifest", "coco_annotations"}, "study")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Unsupported study schema_version; expected integer 1")
    result = {"schema_version": 1,
              "source_manifest": _file(spec["source_manifest"], path.parent, "source_manifest"),
              "coco_annotations": _file(spec["coco_annotations"], path.parent, "coco_annotations"),
              "seeds": _integers(spec.get("seeds", [1, 2, 3, 4, 5]), "seeds", 0, 2),
              "split_seed": _integer(spec.get("split_seed", 20261022), "split_seed", 0, 2**32 - 1),
              "dev_fraction": spec.get("dev_fraction", .2),
              "epochs": _integer(spec.get("epochs", 30), "epochs", 1),
              "budgets": _integers(spec.get("budgets", [10, 25, 50, 100, 200]), "budgets", 1, 1),
              "device": spec.get("device", "cpu"), "comparisons": []}
    fraction = result["dev_fraction"]
    if (type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 < fraction < .5):
        raise ValueError("dev_fraction must be finite and strictly between 0 and 0.5")
    result["dev_fraction"] = float(fraction)
    if result["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if 100 not in result["budgets"]:
        raise ValueError("budgets must include 100, the predeclared local-threshold tuning budget")
    if "config" in spec:
        result["config"] = _file(spec["config"], path.parent, "config")
        config = _read_json(result["config"])
        if not isinstance(config, dict):
            raise ValueError("config must contain a JSON object")
        # Reuse the public solver's validation, without importing torch to plan.
        from .solver import SolverConfig
        SolverConfig(**config.get("solver", config))
        if "solver" in config:
            _object(config, {"solver", "model", "training", "note", "experiment"}, set(), "config")
        if "experiment" in config and not isinstance(config["experiment"], dict):
            raise ValueError("config.experiment must be an object containing descriptive metadata")
        model = config.get("model", {})
        training = config.get("training", {})
        _object(model, _MODEL_FIELDS, set(), "config.model")
        _object(training, _TRAINING_FIELDS, set(), "config.training")
        for key, value in model.items():
            if key in {"use_appearance", "use_rescue", "rescue_same_class"}:
                if type(value) is not bool:
                    raise ValueError(f"config.model.{key} must be boolean")
            elif key in {"overlap_threshold", "distance_gate"}:
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"config.model.{key} must be finite and nonnegative")
                if key == "overlap_threshold" and value > 1:
                    raise ValueError("config.model.overlap_threshold must be at most 1")
            else:
                _integer(value, f"config.model.{key}", 0 if key == "rescue_k" else 1)
        if model.get("hidden_dim", 160) % model.get("heads", 4):
            raise ValueError("config.model.hidden_dim must be divisible by heads")
        for key, value in training.items():
            if key == "num_classes":
                _integer(value, "config.training.num_classes", 1)
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"config.training.{key} must be finite and nonnegative")
            elif key == "assignment_iou" and not 0 < value <= 1:
                raise ValueError("config.training.assignment_iou must be in (0, 1]")
            elif key in {"learning_rate", "gradient_clip"} and value <= 0:
                raise ValueError(f"config.training.{key} must be positive")
    comparisons = spec.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list of predeclared comparisons")
    comparison_keys = {"method_a", "method_b", "budget", "metric", "subset", "repeats", "seed"}
    seen = set()
    for index, item in enumerate(comparisons):
        _object(item, comparison_keys, {"metric"}, f"comparison {index}")
        comparison = {"method_a": item.get("method_a", "dialdet"),
                      "method_b": item.get("method_b", "local"),
                      "budget": _integer(item.get("budget", 100), "comparison budget", 1),
                      "metric": item["metric"], "subset": item.get("subset", "all"),
                      "repeats": _integer(item.get("repeats", 2000), "comparison repeats", 2),
                      "seed": _integer(item.get("seed", 1), "comparison seed", 0, 2**32 - 1)}
        if (comparison["method_a"] not in _METHODS or comparison["method_b"] not in _METHODS
                or comparison["method_a"] == comparison["method_b"]):
            raise ValueError("Comparison methods must be two different supported methods")
        if comparison["metric"] not in _METRICS:
            raise ValueError(f"Unsupported comparison metric: {comparison['metric']}")
        if comparison["subset"] not in ("all", "high_pressure"):
            raise ValueError("comparison subset must be all or high_pressure")
        if comparison["budget"] not in result["budgets"]:
            raise ValueError("Comparison budget must be present in the study budgets")
        identity = tuple(comparison[k] for k in ("method_a", "method_b", "budget", "metric", "subset"))
        if identity in seen:
            raise ValueError("Duplicate predeclared comparison")
        seen.add(identity)
        result["comparisons"].append(comparison)
    return path, result


def _input_snapshot(specification, spec, output):
    """Validate records one at a time; do not retain the COCO feature population."""
    manifest = Path(spec["source_manifest"])
    document = manifest_document(manifest)
    if document["dataset"] != "coco":
        raise ValueError("The study currently supports COCO detection manifests only")
    if len(document["records"]) < 6:
        raise ValueError("Two-fold fitting, development and audit require at least six images")
    if any(entry["role"] != "pool" for entry in document["records"]):
        raise ValueError("Public-checkpoint study source records must all have role=pool")
    if document.get("risk_eligible") is True:
        raise ValueError("This public-checkpoint study cannot consume a risk-eligible protocol")
    files = {str(specification), str(manifest), spec["coco_annotations"]}
    if spec.get("config"):
        files.add(spec["config"])
    files.update(str((manifest.parent / entry["path"]).resolve()) for entry in document["records"])
    for name in files:
        file = Path(name)
        if output == file or output in file.parents:
            raise ValueError(f"Study output overlaps a protected input: {file}")
    runner = Path(__file__).resolve().parent.parent / "run_dialdet_arrhenius.py"
    if output == runner.parent or output in runner.parents:
        raise ValueError("Study output cannot replace or contain the implementation directory")
    records, identities, feature_dims = {}, [], set()
    synthetic = classify_synthetic(document)
    for filename in sorted(files):
        file = Path(filename)
        records[filename] = {"sha256": sha256_file(file), "bytes": file.stat().st_size}
    for entry in document["records"]:
        source = (manifest.parent / entry["path"]).resolve()
        file_hash = records[str(source)]["sha256"]
        if entry.get("sha256") and entry["sha256"] != file_hash:
            raise ValueError(f"Candidate export hash mismatch: {source}")
        record = load_record(source)
        if record.image_id != str(entry["image_id"]):
            raise ValueError("Manifest/record image-ID mismatch")
        # Match load_manifest's semantic interface before checking a declared
        # semantic hash; dataset is supplied by the manifest when absent in NPZ.
        record.metadata = {**record.metadata, "role": entry["role"], "dataset": document["dataset"],
                           "protocol": document.get("protocol", "unspecified"), "source_path": str(source)}
        if entry.get("numerical_sha256") and entry["numerical_sha256"] != record_hash(record):
            raise ValueError(f"Candidate numerical_sha256 mismatch: {record.image_id}")
        if entry.get("semantic_sha256"):
            from .provenance import candidate_semantic_hash
            if entry["semantic_sha256"] != candidate_semantic_hash(record):
                raise ValueError(f"Candidate semantic_sha256 mismatch: {record.image_id}")
        if classify_synthetic(document, [record]) is True:
            synthetic = True
        feature_dims.add(record.features.shape[1])
        try:
            image_id = int(record.image_id)
        except (ValueError, TypeError) as exc:
            raise ValueError("COCO study image IDs must be canonical integers") from exc
        if str(image_id) != record.image_id:
            raise ValueError("COCO study image IDs must be canonical integers")
        identities.append(image_id)
    if len(feature_dims) != 1:
        raise ValueError("All study records must share one frozen feature dimension")
    annotations = _read_json(spec["coco_annotations"])
    if not isinstance(annotations, dict) or not all(isinstance(annotations.get(k), list)
                                                  for k in ("images", "annotations", "categories")):
        raise ValueError("coco_annotations must contain original COCO images, annotations and categories")
    image_ids = [entry.get("id") for entry in annotations["images"]]
    if (any(type(image_id) is not int for image_id in image_ids)
            or len(set(image_ids)) != len(image_ids)):
        raise ValueError("COCO annotation image IDs must be unique integers")
    if not set(identities) <= set(image_ids):
        raise ValueError("COCO annotations do not cover the complete frozen candidate population")
    return records, {"images": len(identities), "image_ids_sha256": _digest(sorted(identities)),
                     "feature_dim": feature_dims.pop(), "is_synthetic": synthetic}


def _make_plan(specification, spec, output, inputs, population):
    # The CLI is imported lazily to avoid a circular dependency with its study handler.
    from .cli import make_parser
    parser = make_parser()
    runner = Path(__file__).resolve().parent.parent / "run_dialdet_arrhenius.py"
    prefix = [sys.executable, str(runner)] if runner.is_file() else [sys.executable, "-m", "dialdet"]
    tasks = []

    def task(identifier, command, destination, options, dependencies, required):
        argv = prefix + [command, "--output", str(destination)] + [str(value) for value in options]
        arguments = vars(parser.parse_args(argv[len(prefix):]))
        tasks.append({"id": identifier, "command": command, "argv": argv,
                      "arguments": arguments, "command_sha256": _digest(argv),
                      "output": str(destination), "dependencies": dependencies,
                      "required_artifacts": required})
        return identifier

    folds = output / "folds"
    fold_paths = [str(folds / f"fold_{fold}.json") for fold in range(2)]
    task("crossfit", "make-crossfit", folds,
         ["--manifest", spec["source_manifest"], "--dataset", "coco", "--seed", spec["split_seed"],
          "--dev-fraction", spec["dev_fraction"]], [], ["fold_0.json", "fold_1.json"])
    evaluations, stats_runs = [], []
    for seed in spec["seeds"]:
        root = output / f"seed_{seed}"
        pipelines = []
        for fold, manifest in enumerate(fold_paths):
            options = ["--manifest", manifest, "--dataset", "coco", "--device", spec["device"],
                       "--seed", seed, "--epochs", spec["epochs"], "--quality-mode", "coco",
                       "--coco-annotations", spec["coco_annotations"],
                       "--development-annotations", spec["coco_annotations"],
                       "--budgets", *spec["budgets"]]
            if spec.get("config"):
                options += ["--config", spec["config"]]
            pipelines.append(task(f"seed_{seed}_fold_{fold}", "pipeline", root / f"fold_{fold}",
                                  options, ["crossfit"],
                                  ["PIPELINE.json", "inference/predictions.jsonl",
                                   "evaluation/EVALUATION.json", "evaluation/per_image_audits.json"]))
        union = root / "union"
        merge = task(f"seed_{seed}_merge", "merge-crossfit", union,
                     ["--dataset", "coco", "--manifests", *fold_paths, "--prediction-dirs",
                      root / "fold_0" / "inference", root / "fold_1" / "inference"],
                     pipelines, ["MERGE.json", "manifest.json", "predictions.jsonl"])
        evaluation = root / "evaluation"
        evaluations.append(task(f"seed_{seed}_evaluate", "evaluate", evaluation,
                                ["--dataset", "coco", "--manifest", union / "manifest.json",
                                 "--predictions", union, "--coco-annotations", spec["coco_annotations"],
                                 "--role", "audit"], [merge],
                                ["EVALUATION.json", "per_image_audits.json"]))
        stats_runs.append({"seed": seed, "audits": str(evaluation / "per_image_audits.json"),
                           "predictions": str(union / "predictions.jsonl"),
                           "run_manifest": str(evaluation / "RUN_MANIFEST.json")})
    generated_specs = {}
    for index, comparison in enumerate(spec["comparisons"]):
        identifier = f"comparison_{index:02d}"
        path = output / "specifications" / f"{identifier}.json"
        generated_specs[str(path)] = {"schema_version": 1, **comparison,
                                      "coco_annotations": spec["coco_annotations"], "runs": stats_runs}
        task(identifier, "paired-statistics", output / "statistics" / identifier,
             ["--specification", path], evaluations.copy(), ["PAIRED_STATISTICS.json"])
    plan = {"schema": PLAN_SCHEMA, "specification": str(specification), "settings": spec,
            "output": str(output), "source": source_fingerprint(), "inputs": inputs,
            "population": population, "tasks": tasks, "generated_specifications": generated_specs,
            "protocol": "public_checkpoint_crossfit", "risk_eligible": False,
            "ap_aggregation": "Evaluate each seed's merged held-out union; never average fold AP.",
            "scope": "Frozen public detector candidates; encoder seeds; no detector retraining or paper-result claim."}
    plan["plan_sha256"] = _digest(plan)
    return plan


def _task_inputs(task):
    result = {}
    for name, filename in invocation_inputs(task["arguments"]).items():
        path = Path(filename).resolve()
        if not path.is_file():
            raise ValueError(f"Missing input for task {task['id']}: {name}: {path}")
        result[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _verify_completed(task, state, plan, dependencies):
    root = Path(task["output"])
    record = _read_json(root / "RUN_MANIFEST.json")
    verified = verify_run(root, check_inputs=True)
    expected_inputs = _task_inputs(task)
    failures = []
    if not verified["valid"] or verified["unlisted_files"]:
        failures.append(f"run artifacts: {verified['errors']}; unlisted={verified['unlisted_files']}")
    if record.get("source", {}).get("sha256") != plan["source"]["sha256"]:
        failures.append("implementation source changed")
    if record.get("command") != task["command"] or record.get("arguments") != task["arguments"]:
        failures.append("recorded CLI command or arguments differ")
    if record.get("inputs") != expected_inputs or state.get("inputs") != expected_inputs:
        failures.append("recorded task inputs differ")
    if state.get("dependencies") != dependencies:
        failures.append("dependency run identities differ")
    if state.get("command_sha256") != task["command_sha256"]:
        failures.append("task command signature differs")
    if state.get("run_sha256") != verified["run_sha256"]:
        failures.append("completed run identity differs")
    missing = set(task["required_artifacts"]) - set(record.get("artifacts", {}))
    if missing:
        failures.append(f"required artifacts are absent from the run record: {sorted(missing)}")
    if failures:
        raise ValueError(f"Cannot trust completed task {task['id']}: {'; '.join(failures)}")
    return verified["run_sha256"]


@contextmanager
def _exclusive_study(output):
    # Arrhenius and the supported shell launcher are POSIX. flock releases even
    # after interruption; a persistent empty lock file does not mean a stale job.
    import fcntl
    with open(output / "STUDY_LOCK", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another study process is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _assert_unchanged(plan, stamps):
    if source_fingerprint()["sha256"] != plan["source"]["sha256"]:
        raise ValueError("Implementation changed during study execution; use a new study output")
    for filename, expected in plan["inputs"].items():
        path = Path(filename)
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        if current != stamps[filename]:
            if stat.st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise ValueError(f"Study input changed during execution: {filename}")
            stamps[filename] = current


def _reject_output_symlinks(output):
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Study outputs cannot contain symlinks: {path}")


def _summary(plan, status, executed):
    tasks = status["tasks"]
    return {"schema": "dialdet-coco-study-result-v1", "status": status["status"],
            "plan_sha256": plan["plan_sha256"], "executed": executed,
            "is_synthetic": plan["population"]["is_synthetic"], "risk_eligible": False,
            "protocol": plan["protocol"], "seeds": plan["settings"]["seeds"],
            "images": plan["population"]["images"], "tasks_total": len(tasks),
            "tasks_completed": sum(item["status"] == "completed" for item in tasks.values()),
            "tasks": {key: item["status"] for key, item in tasks.items()},
            "plan": str(Path(plan["output"]) / "STUDY_PLAN.json"),
            "study_status": str(Path(plan["output"]) / "STUDY_STATUS.json"),
            "note": "Plan and execution integrity do not establish scientific replication or comparative performance."}


def run_study(specification, output, execute=False, resume=False):
    """Plan or execute a fixed two-fold COCO study, with verified task resume.

    ``resume`` permits skipping completed, verified tasks. It never silently
    resumes an incomplete training directory or changes a registered study.
    Relative input paths are resolved against ``specification``; all planned
    child commands use absolute paths and ``sys.executable`` without a shell.
    An installed package without the source-tree runner uses ``-m dialdet``.
    """
    specification, spec = _normalize(specification)
    output = Path(output).resolve()
    inputs, population = _input_snapshot(specification, spec, output)
    plan = _make_plan(specification, spec, output, inputs, population)
    output.mkdir(parents=True, exist_ok=True)
    _reject_output_symlinks(output)
    with _exclusive_study(output):
        plan_path, status_path = output / "STUDY_PLAN.json", output / "STUDY_STATUS.json"
        if plan_path.exists():
            prior = _read_json(plan_path)
            if prior != plan:
                raise ValueError("Study specification, source, input contents, command or output changed; use a new study output")
            if not status_path.is_file():
                raise ValueError("Existing study plan has no status record; use a new study output")
            status = _read_json(status_path)
            expected_ids = {task["id"] for task in plan["tasks"]}
            if (status.get("schema") != STATUS_SCHEMA or status.get("plan_sha256") != plan["plan_sha256"]
                    or set(status.get("tasks", {})) != expected_ids):
                raise ValueError("Study status does not match the registered plan")
            if any(item.get("status") not in {"pending", "running", "failed", "completed"}
                   for item in status["tasks"].values()):
                raise ValueError("Study contains an unknown task status")
            if (status.get("status") not in {"planned", "running", "failed", "completed"}
                    or (status["status"] == "completed"
                        and any(item["status"] != "completed" for item in status["tasks"].values()))):
                raise ValueError("Study summary status is inconsistent with its tasks")
        else:
            unknown = {item.name for item in output.iterdir()} - _CONTROL_FILES
            if unknown:
                raise ValueError(f"New study output must be empty apart from launcher status files: {sorted(unknown)}")
            status = {"schema": STATUS_SCHEMA, "plan_sha256": plan["plan_sha256"], "status": "planned",
                      "tasks": {task["id"]: {"status": "pending"} for task in plan["tasks"]}}
            atomic_json(plan_path, plan)
            atomic_json(status_path, status)
        run_hashes = {}
        # Verify every completed task before launching anything, including tasks
        # which are not the direct input to the next pending command.
        for task in plan["tasks"]:
            state = status["tasks"][task["id"]]
            if state["status"] == "completed":
                if any(dep not in run_hashes for dep in task["dependencies"]):
                    raise ValueError(f"Completed task {task['id']} has an incomplete dependency")
                dependencies = {dep: run_hashes[dep] for dep in task["dependencies"]}
                run_hashes[task["id"]] = _verify_completed(task, state, plan, dependencies)
        if not execute:
            return _summary(plan, status, False)
        started = any(item["status"] != "pending" for item in status["tasks"].values())
        if started and not resume:
            raise ValueError("Study execution already started; use --resume to verify and reuse completed tasks")
        stamps = {name: (Path(name).stat().st_size, Path(name).stat().st_mtime_ns) for name in inputs}
        for filename, document in plan["generated_specifications"].items():
            path = Path(filename)
            if path.exists():
                if _read_json(path) != document:
                    raise ValueError(f"Generated paired-statistics specification changed: {path}")
            else:
                atomic_json(path, document)
        for task in plan["tasks"]:
            identifier = task["id"]
            if identifier in run_hashes:
                continue
            destination = Path(task["output"])
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise ValueError(f"Incomplete task {identifier} has existing outputs at {destination}. "
                                 "Move that failed task directory outside the study, then use --execute --resume; "
                                 "this runner does not adopt or overwrite partial training artifacts.")
            dependencies = {dep: run_hashes[dep] for dep in task["dependencies"]}
            _assert_unchanged(plan, stamps)
            state = {"status": "running", "command_sha256": task["command_sha256"],
                     "dependencies": dependencies, "inputs": _task_inputs(task)}
            status["tasks"][identifier] = state
            status["status"] = "running"
            atomic_json(status_path, status)
            logs = output / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"{identifier}.log"
            try:
                with open(log_path, "w") as log:
                    result = subprocess.run(task["argv"], cwd=str(Path(__file__).resolve().parent.parent),
                                            stdout=log, stderr=subprocess.STDOUT, check=False, shell=False)
                if result.returncode != 0:
                    raise RuntimeError(f"Task {identifier} exited {result.returncode}; see {log_path}")
                _reject_output_symlinks(output)
                _assert_unchanged(plan, stamps)
                record = _read_json(destination / "RUN_MANIFEST.json")
                state["run_sha256"] = record.get("run_sha256")
                run_hashes[identifier] = _verify_completed(task, state, plan, dependencies)
                state["status"] = "completed"
                state["log"] = str(log_path)
                atomic_json(status_path, status)
            except BaseException as exc:
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}", log=str(log_path))
                status["status"] = "failed"
                atomic_json(status_path, status)
                raise
        status["status"] = "completed"
        atomic_json(status_path, status)
        return _summary(plan, status, True)
