"""
ecdna_bench.config — configuration loader for the entire repository.

This module is the one thing every other module depends on. It loads
`configs/default.yaml`, optionally merges `configs/paths.local.yaml` over it,
and returns a typed `Config` object that exposes every knob described in
the YAML as a Python attribute.

Design choices
--------------
1. Dataclasses, not Hydra. One hand-rolled loader is ~300 lines and transparent.
   Hydra brings composability we do not need and a learning curve reviewers
   should not be asked to climb.
2. Extendable: each sub-config dataclass has a `from_dict` classmethod that
   silently drops unknown keys. When a later phase adds a new sub-section
   to the YAML, only the relevant sub-config dataclass needs to grow; this
   file does not have to be rewritten.
3. No post-load mutation. Once loaded, a `Config` is a snapshot of the run.
   If you want a variant (e.g., AND matching instead of OR), create a fresh
   Config by loading a different YAML or by calling `config.replace(...)`.
4. Paths are resolved at load time. Every path becomes an absolute
   `pathlib.Path`. Missing required paths raise a clear error.

Usage
-----
>>> from ecdna_bench import load_config
>>> cfg = load_config("configs/default.yaml")
>>> cfg.evaluation.matching.d_max
20
>>> cfg.paths.data_root
PosixPath('/data/ecdna_fish_dataset')
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


# ==============================================================================
# Small helpers
# ==============================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`, returning a new dict.

    - Dict values are merged key-by-key.
    - Non-dict values in `override` replace those in `base`.
    - Keys in `override` that are absent from `base` are added.

    Neither input is mutated.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _filter_known_fields(cls: type, d: dict | None) -> dict:
    """
    Return a subset of `d` containing only keys matching a field name on
    `cls`. Unknown keys are dropped, which keeps `from_dict` forward-compatible
    with YAML files that carry keys for other, unrelated sub-sections.
    """
    if d is None:
        return {}
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in known}


def _resolve_path(p: str | Path | None, root: Path) -> Path | None:
    """
    Resolve a path relative to the repository root (the directory that
    contains `configs/default.yaml`). Absolute paths are returned unchanged.
    `None` passes through.
    """
    if p is None:
        return None
    p = Path(p)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


# ==============================================================================
# Sub-configs (one dataclass per top-level YAML section)
# ==============================================================================


@dataclass
class SplitsConfig:
    train: Path | None = None
    val: Path | None = None
    test: Path | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "SplitsConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class PathsConfig:
    data_root: Path | None = None
    metadata_csv: Path | None = None
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    results_root: Path | None = None
    logs_root: Path | None = None
    eccount_checkpoint: Path | None = None
    baseline_predictions_root: Path | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "PathsConfig":
        d = d or {}
        splits = SplitsConfig.from_dict(d.get("splits"))
        rest = _filter_known_fields(cls, d)
        rest["splits"] = splits
        return cls(**rest)


@dataclass
class DatasetConfig:
    cell_lines_full: list[str] = field(default_factory=list)
    cell_lines_benchmark: list[str] = field(default_factory=list)
    benchmark_only: bool = True
    use_roi_mask: bool = True
    require_count_mask_consistent: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "DatasetConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class MatchingConfig:
    policy: str = "OR"
    d_max: float = 20.0
    iou_min: float = 0.1
    alpha: float = 0.5
    big_cost: float = 1.0e6
    min_component_area: int = 3
    connectivity: int = 8

    @classmethod
    def from_dict(cls, d: dict | None) -> "MatchingConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class DensityBinsConfig:
    edges: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "DensityBinsConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class SensitivityConfig:
    d_max_grid: list[float] = field(default_factory=list)
    iou_min_grid: list[float] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SensitivityConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class EvaluationConfig:
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    density_bins: DensityBinsConfig = field(default_factory=DensityBinsConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EvaluationConfig":
        d = d or {}
        return cls(
            matching=MatchingConfig.from_dict(d.get("matching")),
            density_bins=DensityBinsConfig.from_dict(d.get("density_bins")),
            sensitivity=SensitivityConfig.from_dict(d.get("sensitivity")),
        )


@dataclass
class ClassicalConfig:
    frozen_params_json: Path | None = None
    # The nested dicts below are kept as raw dicts on purpose: they are
    # passed as-is to the classical preprocessing and detection registries
    # (which know their own schemas). Keeping them as dicts avoids
    # duplicating every knob of every preprocessing operator here.
    defaults: dict[str, Any] = field(default_factory=dict)
    infer: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ClassicalConfig":
        d = d or {}
        return cls(
            frozen_params_json=d.get("frozen_params_json"),
            defaults=d.get("defaults") or {},
            infer=d.get("infer") or {},
        )


@dataclass
class ClassicalOptStageConfig:
    """Shared schema for stages 1/2/3 of the classical BO."""

    min_sequence_length: int | None = None
    max_sequence_length: int | None = None
    max_images_per_cell_line: int | None = None
    stage1_d_max: float | None = None
    n_iter: int | None = None
    init_points: int | None = None
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict | None) -> "ClassicalOptStageConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class ClassicalOptConfig:
    stage1: ClassicalOptStageConfig = field(default_factory=ClassicalOptStageConfig)
    stage2: ClassicalOptStageConfig = field(default_factory=ClassicalOptStageConfig)
    stage3: ClassicalOptStageConfig = field(default_factory=ClassicalOptStageConfig)
    selection_metric: str = "test_f1"

    @classmethod
    def from_dict(cls, d: dict | None) -> "ClassicalOptConfig":
        d = d or {}
        return cls(
            stage1=ClassicalOptStageConfig.from_dict(d.get("stage1")),
            stage2=ClassicalOptStageConfig.from_dict(d.get("stage2")),
            stage3=ClassicalOptStageConfig.from_dict(d.get("stage3")),
            selection_metric=d.get("selection_metric", "test_f1"),
        )


@dataclass
class EcCountTargetsConfig:
    sigma: float = 1.0
    combine: str = "max"
    normalize: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "EcCountTargetsConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class EcCountModelConfig:
    in_channels: int = 3
    out_channels: int = 1
    base_channels: int = 32
    bilinear: bool = True
    norm_type: str = "group"
    num_groups: int = 8
    dropout_p: float = 0.0
    expected_param_count: int | None = 7849601

    @classmethod
    def from_dict(cls, d: dict | None) -> "EcCountModelConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class EcCountLossConfig:
    pos_weight: float = 20.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    dice_smooth: float = 1.0e-6

    @classmethod
    def from_dict(cls, d: dict | None) -> "EcCountLossConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class EcCountTrainConfig:
    batch_size: int = 2
    num_workers: int = 4
    optimizer: str = "adam"
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    epochs: int = 70
    scheduler: str = "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    seed: int = 42
    amp: bool = False
    best_checkpoint_metric: str = "val_loss"
    best_checkpoint_mode: str = "min"

    @classmethod
    def from_dict(cls, d: dict | None) -> "EcCountTrainConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class EcCountAugConfig:
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    brightness_p: float = 0.2
    brightness_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    @classmethod
    def from_dict(cls, d: dict | None) -> "EcCountAugConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class EcCountPostprocessConfig:
    smooth_sigma: float = 0.5
    reapply_roi: bool = True
    threshold_abs: float = 0.35
    peak_min_distance: int = 2
    nms_min_distance: int = 2
    exclude_border: int = 0
    point_disk_radius: int = 3

    @classmethod
    def from_dict(cls, d: dict | None) -> "EcCountPostprocessConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class EcCountConfig:
    input_size: list[int] = field(default_factory=lambda: [1024, 1224])
    targets: EcCountTargetsConfig = field(default_factory=EcCountTargetsConfig)
    model: EcCountModelConfig = field(default_factory=EcCountModelConfig)
    loss: EcCountLossConfig = field(default_factory=EcCountLossConfig)
    train: EcCountTrainConfig = field(default_factory=EcCountTrainConfig)
    augmentation: EcCountAugConfig = field(default_factory=EcCountAugConfig)
    postprocess: EcCountPostprocessConfig = field(default_factory=EcCountPostprocessConfig)
    point_eval_match_radius: float = 5.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "EcCountConfig":
        d = d or {}
        return cls(
            input_size=d.get("input_size", [1024, 1224]),
            targets=EcCountTargetsConfig.from_dict(d.get("targets")),
            model=EcCountModelConfig.from_dict(d.get("model")),
            loss=EcCountLossConfig.from_dict(d.get("loss")),
            train=EcCountTrainConfig.from_dict(d.get("train")),
            augmentation=EcCountAugConfig.from_dict(d.get("augmentation")),
            postprocess=EcCountPostprocessConfig.from_dict(d.get("postprocess")),
            point_eval_match_radius=d.get("point_eval_match_radius", 5.0),
        )


@dataclass
class BenchmarkConfig:
    models: list[str] = field(default_factory=list)
    supplementary_models: list[str] = field(default_factory=list)
    matching_modes: list[str] = field(default_factory=lambda: ["OR", "AND"])
    subset_for_main_aggregate: str = "test"
    subset_for_stratified: str = "all"

    @classmethod
    def from_dict(cls, d: dict | None) -> "BenchmarkConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class FiguresConfig:
    output_formats: list[str] = field(default_factory=lambda: ["pdf", "png"])
    dpi: int = 300
    style: str = "nature"
    font_family: str = "Arial"
    font_size_pt: float = 7.0
    max_height_mm: float = 210.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "FiguresConfig":
        return cls(**_filter_known_fields(cls, d))


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def from_dict(cls, d: dict | None) -> "LoggingConfig":
        return cls(**_filter_known_fields(cls, d))


# ==============================================================================
# Top-level Config
# ==============================================================================


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    classical_opt: ClassicalOptConfig = field(default_factory=ClassicalOptConfig)
    eccount: EcCountConfig = field(default_factory=EcCountConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    figures: FiguresConfig = field(default_factory=FiguresConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Where this config was loaded from — useful for provenance logging.
    source_yaml: Path | None = field(default=None, compare=False)
    local_override_yaml: Path | None = field(default=None, compare=False)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Config":
        d = d or {}
        return cls(
            paths=PathsConfig.from_dict(d.get("paths")),
            dataset=DatasetConfig.from_dict(d.get("dataset")),
            evaluation=EvaluationConfig.from_dict(d.get("evaluation")),
            classical=ClassicalConfig.from_dict(d.get("classical")),
            classical_opt=ClassicalOptConfig.from_dict(d.get("classical_opt")),
            eccount=EcCountConfig.from_dict(d.get("eccount")),
            benchmark=BenchmarkConfig.from_dict(d.get("benchmark")),
            figures=FiguresConfig.from_dict(d.get("figures")),
            logging=LoggingConfig.from_dict(d.get("logging")),
        )

    def replace(self, **overrides: Any) -> "Config":
        """Return a modified copy. Supports shallow dataclass replacement only."""
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dict (for provenance logging)."""
        return dataclasses.asdict(self)


# ==============================================================================
# Load / validate
# ==============================================================================


def _find_repo_root(yaml_path: Path) -> Path:
    """
    Walk upward from the config file to find the repository root.

    We define the repo root as the first ancestor directory that contains
    a `pyproject.toml`. Falls back to the config file's parent if no
    `pyproject.toml` is found.
    """
    p = yaml_path.resolve().parent
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return yaml_path.resolve().parent


def _resolve_paths_inplace(cfg: Config, repo_root: Path) -> None:
    """Convert every path string to an absolute Path on `cfg.paths`."""
    p = cfg.paths
    p.data_root = _resolve_path(p.data_root, repo_root)
    p.metadata_csv = _resolve_path(p.metadata_csv, repo_root)
    p.splits.train = _resolve_path(p.splits.train, repo_root)
    p.splits.val = _resolve_path(p.splits.val, repo_root)
    p.splits.test = _resolve_path(p.splits.test, repo_root)
    p.results_root = _resolve_path(p.results_root, repo_root)
    p.logs_root = _resolve_path(p.logs_root, repo_root)
    p.eccount_checkpoint = _resolve_path(p.eccount_checkpoint, repo_root)
    p.baseline_predictions_root = _resolve_path(p.baseline_predictions_root, repo_root)

    fp = cfg.classical.frozen_params_json
    if fp is not None:
        cfg.classical.frozen_params_json = _resolve_path(fp, repo_root)


class ConfigError(ValueError):
    """Raised when a loaded config is internally inconsistent."""


def _validate(cfg: Config) -> None:
    """
    Light-touch validation. We check for internally inconsistent values
    (e.g. IoU outside [0, 1]) and for keys that we know every downstream
    module will dereference. We do NOT check whether paths exist on disk
    — that is the responsibility of the caller at the time of use,
    because some workflows only need a subset of the paths.
    """
    m = cfg.evaluation.matching
    if m.policy not in {"OR", "AND"}:
        raise ConfigError(f"evaluation.matching.policy must be 'OR' or 'AND', got {m.policy!r}")
    if not 0.0 <= m.iou_min <= 1.0:
        raise ConfigError(f"evaluation.matching.iou_min must be in [0, 1], got {m.iou_min}")
    if m.d_max <= 0:
        raise ConfigError(f"evaluation.matching.d_max must be > 0, got {m.d_max}")
    if not 0.0 <= m.alpha <= 1.0:
        raise ConfigError(f"evaluation.matching.alpha must be in [0, 1], got {m.alpha}")
    if m.min_component_area < 1:
        raise ConfigError(
            f"evaluation.matching.min_component_area must be >= 1, got {m.min_component_area}"
        )
    if m.connectivity not in {4, 8}:
        raise ConfigError(f"evaluation.matching.connectivity must be 4 or 8, got {m.connectivity}")

    eb = cfg.evaluation.density_bins
    if len(eb.edges) != len(eb.labels):
        raise ConfigError(
            "evaluation.density_bins.edges and .labels must have the same length "
            f"(got {len(eb.edges)} and {len(eb.labels)})"
        )

    if cfg.eccount.targets.sigma <= 0:
        raise ConfigError(
            f"eccount.targets.sigma must be > 0, got {cfg.eccount.targets.sigma}"
        )
    if cfg.eccount.targets.combine not in {"max", "sum"}:
        raise ConfigError(
            f"eccount.targets.combine must be 'max' or 'sum', got {cfg.eccount.targets.combine!r}"
        )

    pp = cfg.eccount.postprocess
    if not 0.0 <= pp.threshold_abs <= 1.0:
        raise ConfigError(
            f"eccount.postprocess.threshold_abs must be in [0, 1], got {pp.threshold_abs}"
        )


def load_config(
    yaml_path: str | Path = "configs/default.yaml",
    local_override: str | Path | None = "configs/paths.local.yaml",
    *,
    strict: bool = True,
) -> Config:
    """
    Load a YAML config and return a typed `Config` object.

    Parameters
    ----------
    yaml_path
        Path to the master YAML file. Relative paths are resolved against the
        current working directory.
    local_override
        Path to an optional second YAML that is deep-merged over the master
        config. Intended to hold machine-specific paths in
        `configs/paths.local.yaml`. Pass `None` to disable.
    strict
        If True (default), unknown top-level sections in the YAML raise an
        error. Sub-section keys are always silently ignored if unknown (this
        is what makes the loader forward-compatible).

    Returns
    -------
    Config

    Raises
    ------
    FileNotFoundError
        If `yaml_path` does not exist.
    ConfigError
        If the loaded config is internally inconsistent.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    repo_root = _find_repo_root(yaml_path)

    with open(yaml_path) as f:
        base: dict = yaml.safe_load(f) or {}

    local_path: Path | None = None
    if local_override is not None:
        local_path_candidate = Path(local_override)
        if not local_path_candidate.is_absolute():
            local_path_candidate = (repo_root / local_path_candidate).resolve()
        if local_path_candidate.exists():
            with open(local_path_candidate) as f:
                overrides: dict = yaml.safe_load(f) or {}
            base = _deep_merge(base, overrides)
            local_path = local_path_candidate

    if strict:
        known_top_level = {
            "paths",
            "dataset",
            "evaluation",
            "classical",
            "classical_opt",
            "eccount",
            "benchmark",
            "figures",
            "logging",
        }
        unknown = set(base.keys()) - known_top_level
        if unknown:
            raise ConfigError(f"Unknown top-level YAML keys: {sorted(unknown)}")

    cfg = Config.from_dict(base)
    cfg.source_yaml = yaml_path.resolve()
    cfg.local_override_yaml = local_path

    _resolve_paths_inplace(cfg, repo_root)
    _validate(cfg)
    return cfg
