import inspect
import math
from collections.abc import Mapping
from typing import Any, Callable, List, Optional

import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from models.ng_autoencoder import NGAutoEncoder
from utils.intrinsic_replay import IntrinsicReplay


class NeurogenesisTrainer:
    """
    Orchestrates sequential class learning with neurogenesis.
    Stores per-class, per-layer reconstruction-error history for analysis.
    """

    def __init__(
        self,
        ae: NGAutoEncoder,
        ir: IntrinsicReplay,
        thresholds: List[float],
        max_nodes: List[int],
        max_outliers: float,
        base_lr: float = 1e-3,
        plasticity_epochs: int = 5,
        stability_epochs: int = 2,
        next_layer_epochs: int = 1,
        factor_max_new_nodes: float = 0.1,
        factor_new_nodes: float = 0.1,
        logger: MLFlowLogger = None,
        mean_layer_losses=None,
        early_stop_cfg: dict = None,
        objective_mode: str = "paper_level_ae",
        plasticity_decoder_lr_ratio: float = 0.01,
        stability_lr_ratio: float = 0.01,
        next_layer_lr_ratio: float = 0.01,
        next_layer_optimization: str = "broad",
        replay_old_limit: int | None = None,
        stability_replay_mode: str = "ratio",
        stability_replay_ratio: float = 1.0,
        stability_replay_ratio_base: float = 1.0,
        stability_replay_ratio_max: float = 4.0,
        stability_replay_balanced_max_ratio: float = 4.0,
        stability_replay_per_class_ratio: float = 1.0,
        stability_replay_class_weights: dict | None = None,
        stability_replay_loss_weight: float = 1.0,
        stability_schedule: str = "mixed",
        stability_current_epochs_ratio: float = 1.0,
        stability_replay_epochs_ratio: float = 1.0,
        growth_mode: str = "proportional",
        growth_mode_by_level: dict | None = None,
        absolute_new_nodes: int = 1,
        absolute_new_nodes_by_level: dict | None = None,
        factor_new_nodes_by_level: dict | None = None,
        factor_max_new_nodes_by_level: dict | None = None,
        shape_pressure_mode: str = "none",
        shape_target_ratio: float = 1.0,
        shape_target_ratio_by_level: dict | None = None,
        shape_min_growth_scale: float = 0.25,
        shape_growth_scale_power: float = 1.0,
        shape_gate_power: float = 1.0,
        shape_max_gate_multiplier: float = 10.0,
        max_outliers_by_level: dict | None = None,
        max_outlier_fraction_by_level: dict | None = None,
        global_coupling_cfg: dict | None = None,
        outlier_criterion_diagnostics: dict | None = None,
        quality_growth_gate: dict | None = None,
        adaptive_outlier_threshold: dict | None = None,
    ):
        self.ae = ae
        self.ir = ir
        self.thresholds = thresholds
        self.max_nodes = max_nodes
        self.max_outliers = max_outliers
        self.max_outliers_by_level = self._plain_mapping(max_outliers_by_level or {})
        self.max_outlier_fraction_by_level = self._plain_mapping(
            max_outlier_fraction_by_level or {}
        )
        self.base_lr = base_lr
        self.logger = logger
        self.factor_max_new_nodes = factor_max_new_nodes
        self.factor_new_nodes = factor_new_nodes
        self.growth_mode = str(growth_mode or "proportional").lower()
        self.growth_mode_by_level = self._plain_mapping(growth_mode_by_level or {})
        self.absolute_new_nodes = int(absolute_new_nodes)
        self.absolute_new_nodes_by_level = self._plain_mapping(
            absolute_new_nodes_by_level or {}
        )
        self.factor_new_nodes_by_level = self._plain_mapping(
            factor_new_nodes_by_level or {}
        )
        self.factor_max_new_nodes_by_level = self._plain_mapping(
            factor_max_new_nodes_by_level or {}
        )
        self.shape_pressure_mode = str(shape_pressure_mode or "none").lower()
        self.shape_target_ratio = float(shape_target_ratio)
        self.shape_target_ratio_by_level = self._plain_mapping(
            shape_target_ratio_by_level or {}
        )
        self.shape_min_growth_scale = float(shape_min_growth_scale)
        self.shape_growth_scale_power = float(shape_growth_scale_power)
        self.shape_gate_power = float(shape_gate_power)
        self.shape_max_gate_multiplier = float(shape_max_gate_multiplier)
        valid_growth_modes = {"proportional", "absolute"}
        configured_growth_modes = {
            self.growth_mode,
            *[str(value).lower() for value in self.growth_mode_by_level.values()],
        }
        unknown_growth_modes = configured_growth_modes - valid_growth_modes
        if unknown_growth_modes:
            raise ValueError(
                "Unknown neurogenesis growth_mode "
                f"{sorted(unknown_growth_modes)}. Expected one of "
                f"{sorted(valid_growth_modes)}."
            )
        valid_shape_pressure_modes = {"none", "scale_growth", "scale_gate", "scale_both"}
        if self.shape_pressure_mode not in valid_shape_pressure_modes:
            raise ValueError(
                "Unknown neurogenesis shape_pressure_mode "
                f"'{shape_pressure_mode}'. Expected one of "
                f"{sorted(valid_shape_pressure_modes)}."
            )
        self.mean_layer_losses = mean_layer_losses
        self.early_stop_cfg = early_stop_cfg
        self.plasticity_decoder_lr_ratio = float(plasticity_decoder_lr_ratio)
        self.stability_lr_ratio = float(stability_lr_ratio)
        self.next_layer_lr_ratio = float(next_layer_lr_ratio)
        self.next_layer_optimization = str(next_layer_optimization or "broad").lower()
        if self.next_layer_optimization not in {"broad", "paper_columns"}:
            raise ValueError(
                "Unknown next_layer_optimization "
                f"'{next_layer_optimization}'. Expected 'broad' or 'paper_columns'."
            )
        self.ae.plasticity_decoder_lr_ratio = self.plasticity_decoder_lr_ratio
        self.ae.stability_lr_ratio = self.stability_lr_ratio
        self.ae.next_layer_optimization = self.next_layer_optimization
        self.objective_mode = str(objective_mode or "paper_level_ae").lower()
        valid_objective_modes = {
            "paper_level_ae",
            "paper_local",
            "global_partial",
            "full_reconstruction",
            "local_plasticity_full_stability",
        }
        if self.objective_mode not in valid_objective_modes:
            raise ValueError(
                f"Unknown neurogenesis objective_mode '{objective_mode}'. "
                f"Expected one of {sorted(valid_objective_modes)}."
            )
        self.replay_old_limit = None if replay_old_limit is None else int(replay_old_limit)
        self.stability_replay_mode = str(stability_replay_mode or "ratio").lower()
        valid_replay_modes = {"ratio", "ratio_schedule", "balanced", "only", "paper"}
        if self.stability_replay_mode not in valid_replay_modes:
            raise ValueError(
                f"Unknown stability_replay_mode '{stability_replay_mode}'. "
                f"Expected one of {sorted(valid_replay_modes)}."
            )
        self.stability_replay_ratio = float(stability_replay_ratio)
        self.stability_replay_ratio_base = float(stability_replay_ratio_base)
        self.stability_replay_ratio_max = float(stability_replay_ratio_max)
        self.stability_replay_balanced_max_ratio = float(
            stability_replay_balanced_max_ratio
        )
        self.stability_replay_per_class_ratio = float(stability_replay_per_class_ratio)
        self.stability_replay_class_weights = self._plain_mapping(
            stability_replay_class_weights or {}
        )
        self.stability_replay_loss_weight = float(stability_replay_loss_weight)
        self.stability_schedule = str(stability_schedule or "mixed").lower()
        valid_stability_schedules = {
            "mixed",
            "current_then_replay",
            "replay_then_current",
            "interleave_epochs",
            "interleave_batches",
        }
        if self.stability_schedule not in valid_stability_schedules:
            raise ValueError(
                f"Unknown stability_schedule {stability_schedule!r}. "
                f"Expected one of {sorted(valid_stability_schedules)}."
            )
        self.stability_current_epochs_ratio = float(stability_current_epochs_ratio)
        self.stability_replay_epochs_ratio = float(stability_replay_epochs_ratio)
        self.global_coupling_cfg = self._normalize_global_coupling_cfg(
            global_coupling_cfg or {}
        )
        self.outlier_criterion_diagnostics = self._normalize_outlier_criterion_diagnostics(
            outlier_criterion_diagnostics or {}
        )
        self.quality_growth_gate = self._normalize_quality_growth_gate(
            quality_growth_gate or {}
        )
        self.adaptive_outlier_threshold = self._normalize_adaptive_outlier_threshold(
            adaptive_outlier_threshold or {}
        )
        self._latest_outlier_stats: dict[tuple[str, int, int], dict[str, float]] = {}
        self._phase_loss_history: dict[tuple[Any, int, int, str], list[float]] = {}
        self._recon_eval_batch: torch.Tensor | None = None
        self._replay_counters: dict[str, Any] = {
            "samples": 0,
            "by_class": {},
        }

        # counter for how many classes we've learned so far
        self._class_count = 0

        # epoch settings per phase
        self.plasticity_epochs = plasticity_epochs
        self.stability_epochs = stability_epochs
        self.next_layer_epochs = next_layer_epochs

        # History: class_id -> {'layer_errors': List[List[Tensor]]}
        self.history: dict[Any, dict[str, List[List[Tensor]]]] = {}

    @staticmethod
    def _normalize_outlier_criterion_diagnostics(cfg: Mapping | dict) -> dict[str, Any]:
        source = NeurogenesisTrainer._plain_mapping(cfg or {})
        raw_levels = source.get("levels", [])
        if raw_levels is None:
            levels: set[int] = set()
        elif isinstance(raw_levels, (int, str)):
            levels = {int(raw_levels)}
        else:
            levels = {int(level) for level in raw_levels}
        return {
            "enabled": bool(source.get("enabled", False)),
            "levels": levels,
            "percentiles": [
                float(percentile)
                for percentile in source.get("percentiles", [0.95, 0.975, 0.99])
            ],
        }

    @staticmethod
    def _normalize_quality_growth_gate(cfg: Mapping | dict) -> dict[str, Any]:
        source = NeurogenesisTrainer._plain_mapping(cfg or {})
        raw_levels = source.get("levels", [])
        if raw_levels is None:
            levels: set[int] = set()
        elif isinstance(raw_levels, (int, str)):
            levels = {int(raw_levels)}
        else:
            levels = {int(level) for level in raw_levels}
        min_mean_error = source.get("min_mean_error", None)
        return {
            "enabled": bool(source.get("enabled", False)),
            "levels": levels,
            "threshold_factor": float(source.get("threshold_factor", 1.0)),
            "threshold_factor_by_level": NeurogenesisTrainer._plain_mapping(
                source.get("threshold_factor_by_level", {})
            ),
            "min_mean_error": None
            if min_mean_error is None
            else float(min_mean_error),
            "min_mean_error_by_level": NeurogenesisTrainer._plain_mapping(
                source.get("min_mean_error_by_level", {})
            ),
        }

    @staticmethod
    def _normalize_adaptive_outlier_threshold(cfg: Mapping | dict) -> dict[str, Any]:
        source = NeurogenesisTrainer._plain_mapping(cfg or {})
        raw_levels = source.get("levels", [])
        if raw_levels is None:
            levels: set[int] = set()
        elif isinstance(raw_levels, (int, str)):
            levels = {int(raw_levels)}
        else:
            levels = {int(level) for level in raw_levels}
        percentile = float(source.get("percentile", 0.55))
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("adaptive_outlier_threshold.percentile must be in [0, 1].")
        min_iteration = int(source.get("min_iteration", 1))
        return {
            "enabled": bool(source.get("enabled", False)),
            "levels": levels,
            "percentile": percentile,
            "percentile_by_level": NeurogenesisTrainer._plain_mapping(
                source.get("percentile_by_level", {})
            ),
            "min_iteration": min_iteration,
            "min_iteration_by_level": NeurogenesisTrainer._plain_mapping(
                source.get("min_iteration_by_level", {})
            ),
            "mode": str(source.get("mode", "max_base_quantile") or "max_base_quantile").lower(),
        }

    def _normalize_global_coupling_cfg(self, cfg: Mapping | dict) -> dict[str, Any]:
        source = self._plain_mapping(cfg or {})
        enabled = bool(source.get("enabled", False))
        trigger = str(source.get("trigger", "none") or "none").lower()
        scope = str(source.get("scope", "all") or "all").lower()
        replay = str(source.get("replay", "stability") or "stability").lower()
        valid_triggers = {"none", "after_base", "after_growth_round", "after_level", "after_class"}
        valid_scopes = {"all", "decoder_only", "freeze_old_encoder"}
        valid_replay = {"stability"}
        if trigger not in valid_triggers:
            raise ValueError(
                f"Unknown neurogenesis.global_coupling.trigger '{trigger}'. "
                f"Expected one of {sorted(valid_triggers)}."
            )
        if scope not in valid_scopes:
            raise ValueError(
                f"Unknown neurogenesis.global_coupling.scope '{scope}'. "
                f"Expected one of {sorted(valid_scopes)}."
            )
        if replay not in valid_replay:
            raise ValueError(
                f"Unknown neurogenesis.global_coupling.replay '{replay}'. "
                f"Expected one of {sorted(valid_replay)}."
            )
        early_stop = source.get("early_stop", None)
        if early_stop is not None:
            early_stop = self._plain_mapping(early_stop)
        return {
            "enabled": enabled,
            "trigger": trigger if enabled else "none",
            "epochs": int(source.get("epochs", 5) or 0),
            "lr_ratio": float(source.get("lr_ratio", 0.01) or 0.0),
            "scope": scope,
            "replay": replay,
            "early_stop": early_stop,
        }

    def _log_outlier_metrics(
        self,
        *,
        class_id: Any,
        level: int,
        iteration: int,
        n_outliers: int,
        total_seen: int,
    ) -> None:
        if not self.logger:
            return
        fraction = n_outliers / max(total_seen, 1)
        metric_prefix = f"class_{class_id}"
        metrics = {
            f"{metric_prefix}/level_{level}_n_outliers_round": n_outliers,
            f"{metric_prefix}/level_{level}_outlier_fraction_round": fraction,
        }
        self.logger.log_metrics(metrics, step=iteration)

    def _build_replay_sampler(
        self, device: torch.device, *, n_old_classes: int
    ) -> tuple[Callable[[int], Optional[torch.Tensor]] | None, bool]:
        if self.ir is None or not self.ir.available_classes():
            return None, False
        remaining = self.replay_old_limit
        mode = self.stability_replay_mode
        replay_only = mode == "only"
        old_classes = [int(cls) for cls in self.ir.available_classes()]
        if mode == "ratio_schedule":
            ratio = min(
                self.stability_replay_ratio_base * max(n_old_classes, 1),
                self.stability_replay_ratio_max,
            )
        elif mode == "balanced":
            ratio = min(max(n_old_classes, 1), self.stability_replay_balanced_max_ratio)
        else:
            ratio = self.stability_replay_ratio

        ratio = max(float(ratio), 0.0)

        def _set_remaining(value: int | None) -> None:
            nonlocal remaining
            remaining = value

        def _sample(batch_size: int) -> Optional[torch.Tensor]:
            sync_fn = getattr(self.ir, "sync_encoder_latent_dim", None)
            if callable(sync_fn):
                try:
                    sync_fn()
                except Exception:
                    pass
            nonlocal remaining
            if mode == "paper":
                return self._sample_paper_replay(
                    batch_size,
                    old_classes=old_classes,
                    device=device,
                    remaining_ref=lambda: remaining,
                    update_remaining=_set_remaining,
                )
            take = int(math.ceil(batch_size * ratio))
            if take <= 0:
                return None
            if remaining is not None:
                if remaining <= 0:
                    return None
                take = min(take, remaining)
                remaining -= take
            labels = None
            if hasattr(self.ir, "sample_images_with_labels"):
                replay_flat, labels = self.ir.sample_images_with_labels(None, take)
            else:
                replay_flat = self.ir.sample_images(None, take)
            actual_take = int(replay_flat.size(0))
            self._replay_counters["samples"] = (
                int(self._replay_counters.get("samples", 0)) + actual_take
            )
            if labels is not None:
                by_class = self._replay_counters.setdefault("by_class", {})
                for label in labels.detach().cpu().tolist():
                    cls = int(label)
                    by_class[cls] = int(by_class.get(cls, 0)) + 1
            return replay_flat.to(device, non_blocking=True)

        return _sample, replay_only

    def _sample_paper_replay(
        self,
        batch_size: int,
        *,
        old_classes: list[int],
        device: torch.device,
        remaining_ref: Callable[[], int | None],
        update_remaining: Callable[[int | None], None],
    ) -> Optional[torch.Tensor]:
        """Sample replay examples from every old class, matching the paper description."""
        if not old_classes:
            return None
        per_class = int(
            math.ceil(batch_size * max(self.stability_replay_per_class_ratio, 0.0))
        )
        if per_class <= 0:
            return None
        requested = per_class * len(old_classes)
        remaining = remaining_ref()
        if remaining is not None:
            if remaining <= 0:
                return None
            requested = min(requested, remaining)
            update_remaining(remaining - requested)
        counts = self._weighted_replay_counts(old_classes, requested)

        chunks: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        for cls, take in counts.items():
            if take <= 0:
                continue
            labels = None
            if hasattr(self.ir, "sample_images_with_labels"):
                replay_flat, labels = self.ir.sample_images_with_labels(cls, take)
            else:
                replay_flat = self.ir.sample_images(cls, take)
                labels = torch.full(
                    (int(replay_flat.size(0)),), int(cls), dtype=torch.long
                )
            if replay_flat.numel() == 0:
                continue
            chunks.append(replay_flat)
            if labels is not None:
                labels_all.append(labels.detach().cpu())

        if not chunks:
            return None
        replay = torch.cat(chunks, dim=0)
        actual_take = int(replay.size(0))
        self._replay_counters["samples"] = (
            int(self._replay_counters.get("samples", 0)) + actual_take
        )
        if labels_all:
            by_class = self._replay_counters.setdefault("by_class", {})
            for label in torch.cat(labels_all, dim=0).tolist():
                cls = int(label)
                by_class[cls] = int(by_class.get(cls, 0)) + 1
        return replay.to(device, non_blocking=True)

    def _weighted_replay_counts(
        self, old_classes: list[int], requested: int
    ) -> dict[int, int]:
        if requested <= 0 or not old_classes:
            return {cls: 0 for cls in old_classes}
        weights = {}
        for cls in old_classes:
            raw_weight = self.stability_replay_class_weights.get(
                str(cls), self.stability_replay_class_weights.get(cls, 1.0)
            )
            weights[cls] = max(float(raw_weight), 0.0)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return {cls: 0 for cls in old_classes}

        raw = {cls: requested * weights[cls] / total_weight for cls in old_classes}
        counts = {cls: int(math.floor(raw[cls])) for cls in old_classes}
        assigned = sum(counts.values())
        leftovers = max(int(requested) - assigned, 0)
        order = sorted(old_classes, key=lambda cls: (raw[cls] - counts[cls], -cls), reverse=True)
        for cls in order[:leftovers]:
            counts[cls] += 1
        return counts

    def _build_phase_early_stop_cfg(
        self, level: Optional[int] = None, *, phase: Optional[str] = None
    ) -> Optional[dict]:
        """Return a per-phase early-stop config with optional threshold goal."""
        if not self.early_stop_cfg:
            return None
        source = self._plain_mapping(self.early_stop_cfg)
        cfg = dict(source)
        by_phase = self._plain_mapping(cfg.pop("early_stop_by_phase", {}))
        by_level = self._plain_mapping(cfg.pop("early_stop_by_level", {}))
        by_phase_and_level = self._plain_mapping(
            cfg.pop("early_stop_by_phase_and_level", {})
        )

        phase_key = str(phase or "").lower()
        if phase_key and phase_key in by_phase:
            cfg.update(self._plain_mapping(by_phase[phase_key]))
        if level is not None:
            level_key = str(int(level))
            if level_key in by_level:
                cfg.update(self._plain_mapping(by_level[level_key]))
            phase_level_cfg = self._phase_level_early_stop_cfg(
                by_phase_and_level, phase_key=phase_key, level_key=level_key
            )
            if phase_level_cfg:
                cfg.update(phase_level_cfg)

        use_goal = cfg.pop("use_threshold_goal", False)
        factor = cfg.pop("threshold_goal_factor", 1.0)
        factor_plasticity = cfg.pop("threshold_goal_factor_plasticity", None)
        factor_stability = cfg.pop("threshold_goal_factor_stability", None)
        if phase == "plasticity" and factor_plasticity is not None:
            factor = float(factor_plasticity)
        elif phase == "stability" and factor_stability is not None:
            factor = float(factor_stability)
        if use_goal and self.thresholds:
            idx = len(self.thresholds) - 1 if level is None else max(level, 0)
            idx = min(idx, len(self.thresholds) - 1)
            cfg["goal"] = self.thresholds[idx] * factor
        return cfg

    @classmethod
    def _plain_mapping(cls, value: Any) -> dict:
        """Convert OmegaConf/dict-like config nodes into plain string-key dicts."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            return value
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = cls._plain_mapping(item) if isinstance(item, Mapping) else item
        return out

    @staticmethod
    def _phase_level_early_stop_cfg(
        mapping: dict, *, phase_key: str, level_key: str
    ) -> dict:
        """Read either phase->level or level->phase override maps."""
        if not mapping:
            return {}
        phase_first = mapping.get(phase_key, {})
        if isinstance(phase_first, Mapping) and level_key in phase_first:
            return dict(phase_first[level_key])
        level_first = mapping.get(level_key, {})
        if isinstance(level_first, Mapping) and phase_key in level_first:
            return dict(level_first[phase_key])
        return {}

    def _call_stability_phase(self, **kwargs):
        accepted = inspect.signature(self.ae.stability_phase).parameters
        return self.ae.stability_phase(
            **{key: value for key, value in kwargs.items() if key in accepted}
        )

    def _scheduled_stability_phase(self, **kwargs):
        """Run stability either mixed or as current/replay-only sub-phases."""
        schedule = self.stability_schedule
        if schedule == "mixed" or kwargs.get("old_x") is None:
            hist = self._call_stability_phase(**kwargs)
            hist["_new_epochs"] = len(hist.get("epoch_loss", []))
            return hist

        if schedule == "interleave_batches":
            interleave_kwargs = dict(kwargs)
            interleave_kwargs["replay_interleave_batches"] = True
            hist = self._call_stability_phase(**interleave_kwargs)
            hist["_new_epochs"] = len(hist.get("epoch_loss", []))
            hist["_schedule"] = ["interleave_batches"] * len(hist.get("epoch_loss", []))
            return hist

        base_epochs = int(kwargs.get("epochs", 0))
        current_epochs = max(1, int(math.ceil(base_epochs * self.stability_current_epochs_ratio)))
        replay_epochs = max(1, int(math.ceil(base_epochs * self.stability_replay_epochs_ratio)))

        current_kwargs = dict(kwargs)
        current_kwargs.update({"epochs": current_epochs, "old_x": None, "replay_only": False})
        replay_kwargs = dict(kwargs)
        replay_kwargs.update({"epochs": replay_epochs, "replay_only": True})

        if schedule == "interleave_epochs":
            merged: dict[str, list[float] | int | list[str]] = {
                "epoch_loss": [],
                "_new_epochs": 0,
                "_schedule": [],
            }
            remaining = {"current": current_epochs, "replay": replay_epochs}
            phase_kwargs_by_name = {"current": current_kwargs, "replay": replay_kwargs}
            while remaining["current"] > 0 or remaining["replay"] > 0:
                for phase_name in ("current", "replay"):
                    if remaining[phase_name] <= 0:
                        continue
                    phase_kwargs = dict(phase_kwargs_by_name[phase_name])
                    phase_kwargs["epochs"] = 1
                    hist = self._call_stability_phase(**phase_kwargs)
                    losses = list(hist.get("epoch_loss", []))
                    merged["epoch_loss"].extend(losses)
                    merged["_schedule"].extend([phase_name] * len(losses))
                    if phase_name == "current":
                        merged["_new_epochs"] = int(merged["_new_epochs"]) + len(losses)
                    remaining[phase_name] -= 1
            return merged

        phases = (
            ("current", current_kwargs),
            ("replay", replay_kwargs),
        )
        if schedule == "replay_then_current":
            phases = tuple(reversed(phases))

        merged: dict[str, list[float] | int | list[str]] = {
            "epoch_loss": [],
            "_new_epochs": 0,
            "_schedule": [],
        }
        for phase_name, phase_kwargs in phases:
            hist = self._call_stability_phase(**phase_kwargs)
            losses = list(hist.get("epoch_loss", []))
            merged["epoch_loss"].extend(losses)
            merged["_schedule"].extend([phase_name] * len(losses))
            if phase_name == "current":
                merged["_new_epochs"] = int(merged["_new_epochs"]) + len(losses)
        return merged

    def _phase_forward_fn(self, level: int, phase: str):
        """Return the diagnostic reconstruction objective for a neurogenesis phase."""
        phase = str(phase).lower()
        if self.objective_mode == "full_reconstruction":
            return None
        if (
            self.objective_mode == "local_plasticity_full_stability"
            and phase == "stability"
        ):
            return None
        if self.objective_mode in {"paper_level_ae", "paper_local"}:
            return lambda x, _level=level: self.ae.forward_level_ae(
                x, _level, ret_target=True
            )
        return lambda x, _level=level: self.ae.forward_partial(x, _level)

    def _max_outliers_allowed(self, total_seen: int, level: int | None = None) -> int:
        """Return the growth quota for a level, as an absolute count."""
        if level is not None:
            level_key = str(int(level))
            if level_key in self.max_outliers_by_level:
                return int(self.max_outliers_by_level[level_key])
            if level_key in self.max_outlier_fraction_by_level:
                fraction = float(self.max_outlier_fraction_by_level[level_key])
                return int(math.ceil(max(total_seen, 1) * fraction))
        if self.max_outliers < 1:
            return int(math.ceil(max(total_seen, 1) * float(self.max_outliers)))
        return int(self.max_outliers)

    def _level_override(self, mapping: dict, level: int, default: Any) -> Any:
        if not mapping:
            return default
        level_key = str(int(level))
        return mapping.get(level_key, default)

    def _growth_request(
        self,
        *,
        level: int,
        n_outliers: int,
        nodes_existing: int,
        n_plastic_neurons: int,
        shape_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute requested and capped new-node counts for one growth round."""
        shape_info = shape_info or self._shape_pressure(level)
        mode = str(
            self._level_override(self.growth_mode_by_level, level, self.growth_mode)
        ).lower()
        factor_new = float(
            self._level_override(
                self.factor_new_nodes_by_level, level, self.factor_new_nodes
            )
        )
        factor_max = float(
            self._level_override(
                self.factor_max_new_nodes_by_level, level, self.factor_max_new_nodes
            )
        )
        absolute_new = int(
            self._level_override(
                self.absolute_new_nodes_by_level, level, self.absolute_new_nodes
            )
        )
        if mode == "proportional":
            requested_raw = int(math.ceil(factor_new * int(n_outliers)))
        elif mode == "absolute":
            requested_raw = absolute_new
        else:
            raise ValueError(
                f"Unknown neurogenesis growth_mode '{mode}'. "
                "Expected 'proportional' or 'absolute'."
            )
        requested = requested_raw
        if self.shape_pressure_mode in {"scale_growth", "scale_both"}:
            growth_scale = float(shape_info.get("growth_scale", 1.0))
            if requested_raw > 0:
                requested = max(1, int(math.ceil(requested_raw * growth_scale)))
        per_round_cap = int(math.ceil(factor_max * int(nodes_existing)))
        remaining = int(self.max_nodes[level]) - int(n_plastic_neurons)
        actual = int(min(requested, per_round_cap, remaining))
        return {
            "growth_mode": mode,
            "requested_new_nodes_before_shape": requested_raw,
            "requested_new_nodes": requested,
            "actual_new_nodes": actual,
            "factor_new_nodes_used": factor_new,
            "factor_max_new_nodes_used": factor_max,
            "absolute_new_nodes_used": absolute_new,
            "remaining_new_nodes_before_growth": remaining,
            "per_round_cap": per_round_cap,
            **shape_info,
        }

    def _shape_pressure(self, level: int) -> dict[str, Any]:
        """Return soft funnel-shape pressure for a growth decision."""
        target_ratio = float(
            self._level_override(
                self.shape_target_ratio_by_level, level, self.shape_target_ratio
            )
        )
        base = {
            "shape_pressure_mode": self.shape_pressure_mode,
            "shape_target_ratio": target_ratio,
            "shape_size_ratio": 0.0,
            "shape_over_target": 1.0,
            "growth_scale": 1.0,
            "gate_multiplier": 1.0,
        }
        if (
            self.shape_pressure_mode == "none"
            or level <= 0
            or target_ratio <= 0
            or level >= len(self.ae.hidden_sizes)
        ):
            return base
        prev_size = max(float(self.ae.hidden_sizes[level - 1]), 1.0)
        this_size = float(self.ae.hidden_sizes[level])
        size_ratio = this_size / prev_size
        over_target = max(size_ratio / target_ratio, 1.0)
        growth_scale = 1.0
        gate_multiplier = 1.0
        if over_target > 1.0:
            growth_scale = max(
                self.shape_min_growth_scale,
                over_target ** (-max(self.shape_growth_scale_power, 0.0)),
            )
            gate_multiplier = min(
                self.shape_max_gate_multiplier,
                over_target ** max(self.shape_gate_power, 0.0),
            )
        base.update(
            {
                "shape_size_ratio": size_ratio,
                "shape_over_target": over_target,
                "growth_scale": growth_scale,
                "gate_multiplier": gate_multiplier,
            }
        )
        return base

    def _effective_max_outliers_allowed(
        self, *, level: int, total_seen: int, shape_info: dict[str, Any]
    ) -> int:
        base_allowed = self._max_outliers_allowed(total_seen, level=level)
        if self.shape_pressure_mode not in {"scale_gate", "scale_both"}:
            return base_allowed
        multiplier = max(float(shape_info.get("gate_multiplier", 1.0)), 1.0)
        return int(math.ceil(base_allowed * multiplier))

    def _effective_outlier_threshold(
        self, *, errors: Tensor, level: int, iteration: int
    ) -> dict[str, Any]:
        base_threshold = float(self.thresholds[level])
        cfg = self.adaptive_outlier_threshold
        levels = cfg.get("levels", set())
        enabled = bool(cfg.get("enabled", False)) and (
            not levels or int(level) in levels
        )
        min_iteration = int(
            self._level_override(
                cfg.get("min_iteration_by_level", {}),
                level,
                cfg.get("min_iteration", 1),
            )
        )
        percentile = float(
            self._level_override(
                cfg.get("percentile_by_level", {}),
                level,
                cfg.get("percentile", 0.55),
            )
        )
        percentile = min(max(percentile, 0.0), 1.0)
        active = bool(enabled and int(iteration) >= min_iteration and errors.numel() > 0)
        adaptive_threshold = base_threshold
        if active:
            adaptive_threshold = float(torch.quantile(errors, percentile).item())
        mode = str(cfg.get("mode", "max_base_quantile") or "max_base_quantile").lower()
        if mode == "quantile":
            threshold = adaptive_threshold if active else base_threshold
        elif mode == "max_base_quantile":
            threshold = max(base_threshold, adaptive_threshold) if active else base_threshold
        else:
            raise ValueError(
                "Unknown adaptive_outlier_threshold.mode "
                f"{mode!r}. Expected max_base_quantile or quantile."
            )
        return {
            "enabled": enabled,
            "active": active,
            "threshold": threshold,
            "base_threshold": base_threshold,
            "adaptive_threshold": adaptive_threshold,
            "percentile": percentile,
            "min_iteration": min_iteration,
            "mode": mode,
        }

    def _quality_growth_gate(
        self, *, class_id: Any, level: int, iteration: int
    ) -> dict[str, Any]:
        cfg = self.quality_growth_gate
        levels = cfg.get("levels", set())
        enabled = bool(cfg.get("enabled", False)) and (
            not levels or int(level) in levels
        )
        stats = self._latest_outlier_stats.get(
            (str(class_id), int(level), int(iteration)), {}
        )
        mean_error = float(stats.get("pixel_mean", float("nan")))
        threshold_factor = float(
            self._level_override(
                cfg.get("threshold_factor_by_level", {}),
                level,
                cfg.get("threshold_factor", 1.0),
            )
        )
        min_mean_by_level = cfg.get("min_mean_error_by_level", {})
        configured_min = self._level_override(
            min_mean_by_level, level, cfg.get("min_mean_error", None)
        )
        if configured_min is None:
            required_mean = float(self.thresholds[level]) * threshold_factor
        else:
            required_mean = float(configured_min)
        quality_passes = True
        if enabled:
            quality_passes = bool(math.isfinite(mean_error) and mean_error > required_mean)
        return {
            "enabled": enabled,
            "mean_error": mean_error,
            "required_mean_error": required_mean,
            "threshold_factor": threshold_factor,
            "quality_passes": quality_passes,
        }

    def _log_adaptive_outlier_threshold(
        self,
        *,
        class_id: Any,
        level: int,
        iteration: int,
        total_seen: int,
        threshold_info: dict[str, Any],
    ) -> None:
        if not self.logger:
            return
        prefix = (
            f"diagnostics/adaptive_threshold/class_{class_id}/"
            f"level_{level}/iteration_{iteration}"
        )
        self.logger.log_metrics(
            {
                f"{prefix}/enabled": 1.0 if threshold_info["enabled"] else 0.0,
                f"{prefix}/active": 1.0 if threshold_info["active"] else 0.0,
                f"{prefix}/threshold": float(threshold_info["threshold"]),
                f"{prefix}/base_threshold": float(threshold_info["base_threshold"]),
                f"{prefix}/adaptive_threshold": float(
                    threshold_info["adaptive_threshold"]
                ),
                f"{prefix}/percentile": float(threshold_info["percentile"]),
                f"{prefix}/min_iteration": float(threshold_info["min_iteration"]),
                f"{prefix}/total_seen": float(total_seen),
                f"{prefix}/mode_is_quantile": 1.0
                if threshold_info["mode"] == "quantile"
                else 0.0,
            },
            step=self._class_count,
        )

    def _log_growth_request(
        self,
        *,
        class_id: Any,
        level: int,
        round_idx: int,
        n_outliers: int,
        total_seen: int,
        request: dict[str, Any],
    ) -> None:
        if not self.logger:
            return
        prefix = f"diagnostics/growth_request/class_{class_id}/level_{level}/round_{round_idx}"
        metrics = {
            f"{prefix}/requested_new_nodes_before_shape": float(
                request["requested_new_nodes_before_shape"]
            ),
            f"{prefix}/requested_new_nodes": float(request["requested_new_nodes"]),
            f"{prefix}/actual_new_nodes": float(request["actual_new_nodes"]),
            f"{prefix}/factor_new_nodes_used": float(request["factor_new_nodes_used"]),
            f"{prefix}/factor_max_new_nodes_used": float(request["factor_max_new_nodes_used"]),
            f"{prefix}/absolute_new_nodes_used": float(request["absolute_new_nodes_used"]),
            f"{prefix}/remaining_new_nodes_before_growth": float(
                request["remaining_new_nodes_before_growth"]
            ),
            f"{prefix}/per_round_cap": float(request["per_round_cap"]),
            f"{prefix}/n_outliers": float(n_outliers),
            f"{prefix}/outlier_fraction": float(n_outliers / max(total_seen, 1)),
            f"{prefix}/shape_target_ratio": float(request["shape_target_ratio"]),
            f"{prefix}/shape_size_ratio": float(request["shape_size_ratio"]),
            f"{prefix}/shape_over_target": float(request["shape_over_target"]),
            f"{prefix}/growth_scale": float(request["growth_scale"]),
            f"{prefix}/gate_multiplier": float(request["gate_multiplier"]),
            f"{prefix}/mode_is_absolute": 1.0
            if request["growth_mode"] == "absolute"
            else 0.0,
        }
        self.logger.log_metrics(metrics, step=self._class_count)

    def _log_growth_gate(
        self,
        *,
        class_id: Any,
        level: int,
        round_idx: int,
        n_outliers: int,
        total_seen: int,
        base_allowed: int,
        effective_allowed: int,
        shape_info: dict[str, Any],
        quality_info: dict[str, Any] | None = None,
    ) -> None:
        if not self.logger:
            return
        quality_info = quality_info or {
            "enabled": False,
            "mean_error": float("nan"),
            "required_mean_error": float("nan"),
            "threshold_factor": 1.0,
            "quality_passes": True,
        }
        prefix = f"diagnostics/growth_gate/class_{class_id}/level_{level}/round_{round_idx}"
        self.logger.log_metrics(
            {
                f"{prefix}/n_outliers": float(n_outliers),
                f"{prefix}/outlier_fraction": float(n_outliers / max(total_seen, 1)),
                f"{prefix}/base_allowed_outliers": float(base_allowed),
                f"{prefix}/effective_allowed_outliers": float(effective_allowed),
                f"{prefix}/shape_target_ratio": float(shape_info["shape_target_ratio"]),
                f"{prefix}/shape_size_ratio": float(shape_info["shape_size_ratio"]),
                f"{prefix}/shape_over_target": float(shape_info["shape_over_target"]),
                f"{prefix}/growth_scale": float(shape_info["growth_scale"]),
                f"{prefix}/gate_multiplier": float(shape_info["gate_multiplier"]),
                f"{prefix}/quality_gate_enabled": 1.0
                if bool(quality_info["enabled"])
                else 0.0,
                f"{prefix}/quality_mean_error": float(quality_info["mean_error"]),
                f"{prefix}/quality_required_mean_error": float(
                    quality_info["required_mean_error"]
                ),
                f"{prefix}/quality_threshold_factor": float(
                    quality_info["threshold_factor"]
                ),
                f"{prefix}/quality_passes": 1.0
                if bool(quality_info["quality_passes"])
                else 0.0,
            },
            step=self._class_count,
        )

    def _model_device(self) -> torch.device:
        params = getattr(self.ae, "parameters", None)
        if params is None:
            return torch.device("cpu")
        first = next(params(), None)
        if first is None:
            return torch.device("cpu")
        return first.device

    def _snapshot_replay_counters(self) -> dict[str, Any]:
        return {
            "samples": int(self._replay_counters.get("samples", 0)),
            "by_class": dict(self._replay_counters.get("by_class", {})),
        }

    def _log_replay_delta(
        self,
        *,
        before: dict[str, Any],
        class_id: Any,
        level: int,
        round_idx: int,
        phase: str,
        new_samples: int,
    ) -> None:
        if not self.logger:
            return
        after = self._snapshot_replay_counters()
        replay_samples = int(after["samples"] - before.get("samples", 0))
        metrics = {
            f"diagnostics/replay/class_{class_id}/level_{level}/round_{round_idx}/{phase}_new_samples": float(new_samples),
            f"diagnostics/replay/class_{class_id}/level_{level}/round_{round_idx}/{phase}_replay_samples": float(replay_samples),
        }
        before_by_class = before.get("by_class", {})
        for cls, count in after.get("by_class", {}).items():
            delta = int(count) - int(before_by_class.get(cls, 0))
            if delta:
                metrics[
                    f"diagnostics/replay/class_{class_id}/level_{level}/round_{round_idx}/{phase}_replay_class_{cls}_samples"
                ] = float(delta)
        self.logger.log_metrics(metrics, step=self._class_count)

    def _snapshot_level_params(self, level: int) -> dict[str, list[Tensor]]:
        enc = self.ae._encoder_layer(level)
        dec = self.ae._decoder_layer(level)
        groups = {
            "new_encoder": list(enc.parameters_plastic()),
            "old_encoder": list(enc.parameters_mature()),
            "new_decoder": list(dec.parameters_plastic()),
            "old_decoder": list(dec.parameters_mature()),
        }
        return {
            name: [param.detach().clone() for param in params]
            for name, params in groups.items()
        }

    def _log_param_delta(
        self,
        *,
        before: dict[str, list[Tensor]],
        class_id: Any,
        level: int,
        round_idx: int,
        phase: str,
    ) -> None:
        if not self.logger:
            return
        after = self._snapshot_level_params(level)
        metrics: dict[str, float] = {}
        for group, before_params in before.items():
            after_params = after.get(group, [])
            total_sq = 0.0
            for old, new in zip(before_params, after_params):
                if old.shape != new.shape:
                    continue
                diff = (new.detach().cpu() - old.detach().cpu()).float()
                total_sq += float(torch.sum(diff * diff).item())
            metrics[
                f"diagnostics/param_delta/class_{class_id}/level_{level}/round_{round_idx}/{phase}_{group}_l2"
            ] = math.sqrt(total_sq)
        if metrics:
            self.logger.log_metrics(metrics, step=self._class_count)

    def _snapshot_global_params(self) -> dict[str, list[Tensor]]:
        groups: dict[str, list[Tensor]] = {
            "encoder_mature": [],
            "encoder_plastic": [],
            "decoder_mature": [],
            "decoder_plastic": [],
        }
        for module in self.ae.encoder:
            if hasattr(module, "parameters_mature"):
                groups["encoder_mature"].extend(list(module.parameters_mature()))
            if hasattr(module, "parameters_plastic"):
                groups["encoder_plastic"].extend(list(module.parameters_plastic()))
        for module in self.ae.decoder:
            if hasattr(module, "parameters_mature"):
                groups["decoder_mature"].extend(list(module.parameters_mature()))
            if hasattr(module, "parameters_plastic"):
                groups["decoder_plastic"].extend(list(module.parameters_plastic()))
        return {
            name: [param.detach().clone() for param in params]
            for name, params in groups.items()
        }

    def _log_global_param_delta(
        self,
        *,
        before: dict[str, list[Tensor]] | None,
        class_id: Any,
        trigger: str,
        level: int | None,
        round_idx: int | None,
    ) -> None:
        if not self.logger or before is None:
            return
        after = self._snapshot_global_params()
        metrics: dict[str, float] = {}
        for group, before_params in before.items():
            after_params = after.get(group, [])
            total_sq = 0.0
            for old, new in zip(before_params, after_params):
                if old.shape != new.shape:
                    continue
                diff = (new.detach().cpu() - old.detach().cpu()).float()
                total_sq += float(torch.sum(diff * diff).item())
            key = (
                f"diagnostics/param_delta/class_{class_id}/coupling/{trigger}"
                f"/level_{level if level is not None else 'none'}"
                f"/round_{round_idx if round_idx is not None else 'none'}"
                f"/{group}_l2"
            )
            metrics[key] = math.sqrt(total_sq)
        if metrics:
            self.logger.log_metrics(metrics, step=self._class_count)

    def _global_coupling_enabled(self, trigger: str) -> bool:
        cfg = self.global_coupling_cfg
        return bool(
            cfg.get("enabled", False)
            and cfg.get("trigger") == trigger
            and int(cfg.get("epochs", 0)) > 0
            and float(cfg.get("lr_ratio", 0.0)) > 0
        )

    def _make_global_coupling_optimizer(self, lr: float) -> torch.optim.Optimizer:
        scope = str(self.global_coupling_cfg.get("scope", "all"))
        for param in self.ae.parameters():
            param.requires_grad_(False)

        params: list[Tensor] = []
        for module in self.ae.encoder:
            if scope == "decoder_only":
                continue
            if scope == "freeze_old_encoder":
                if hasattr(module, "parameters_plastic"):
                    params.extend(list(module.parameters_plastic()))
            else:
                params.extend(list(module.parameters()))
        for module in self.ae.decoder:
            params.extend(list(module.parameters()))

        trainable = [param for param in params if param is not None]
        for param in trainable:
            param.requires_grad_(True)
        if not trainable:
            raise RuntimeError(
                f"No trainable parameters for global coupling scope '{scope}'."
            )
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    def _build_coupling_epoch_logger(
        self,
        *,
        class_id: Any,
        trigger: str,
        level: int | None,
        round_idx: int | None,
    ):
        if not self.logger:
            return None
        log_dict_fn = getattr(self.logger, "log_dict", None)
        log_metrics_fn = getattr(self.logger, "log_metrics", None)

        def _callback(epoch_idx: int, summary: dict):
            phase = f"coupling_{trigger}"
            payload = {
                "class_id": class_id,
                "trigger": trigger,
                "level": level,
                "round": round_idx,
                "epoch": epoch_idx + 1,
                **summary,
            }
            if callable(log_dict_fn):
                artifact = (
                    f"coupling/class_{class_id}/{trigger}/"
                    f"level_{level if level is not None else 'none'}/"
                    f"round_{round_idx if round_idx is not None else 'none'}/"
                    f"epoch_{epoch_idx + 1}.json"
                )
                log_dict_fn(payload, artifact)
            if callable(log_metrics_fn) and "loss" in summary:
                metric_base = (
                    f"coupling/class_{class_id}/{trigger}/"
                    f"level_{level if level is not None else 'none'}/"
                    f"round_{round_idx if round_idx is not None else 'none'}"
                )
                log_metrics_fn({f"{metric_base}_loss": float(summary["loss"])}, step=epoch_idx + 1)
                if "eval_loss" in summary:
                    log_metrics_fn(
                        {f"{metric_base}_eval_loss": float(summary["eval_loss"])},
                        step=epoch_idx + 1,
                    )
                key = (class_id, -1 if level is None else level, round_idx or 0, phase)
                self._phase_loss_history.setdefault(key, []).append(float(summary["loss"]))

        return _callback

    def _run_global_coupling(
        self,
        *,
        class_id: Any,
        loader: DataLoader,
        replay_sampler: Callable[[int], Optional[torch.Tensor]] | None,
        replay_only: bool,
        trigger: str,
        level: int | None = None,
        round_idx: int | None = None,
    ) -> None:
        if not self._global_coupling_enabled(trigger):
            return
        cfg = self.global_coupling_cfg
        lr = self.base_lr * float(cfg.get("lr_ratio", 0.01))
        epochs = int(cfg.get("epochs", 0))
        opt = self._make_global_coupling_optimizer(lr)
        early_stop_cfg = cfg.get("early_stop")
        early_stopper = None
        if early_stop_cfg:
            from models.ng_autoencoder import EarlyStopper

            early_stopper = EarlyStopper(**early_stop_cfg)
        before_params = self._snapshot_global_params() if self.logger else None
        before_replay = self._snapshot_replay_counters()
        try:
            hist = self.ae._run_epoch_loop(
                loader,
                opt,
                epochs,
                replay=replay_sampler,
                replay_only=replay_only,
                eval_batch=self._recon_eval_batch,
                early_stop_on_eval=self._recon_eval_batch is not None,
                early_stopper=early_stopper,
                forward_fn=None,
                epoch_logger=self._build_coupling_epoch_logger(
                    class_id=class_id,
                    trigger=trigger,
                    level=level,
                    round_idx=round_idx,
                ),
                loop_label=f"coupling_{trigger}",
                replay_loss_weight=self.stability_replay_loss_weight,
            )
        finally:
            for param in self.ae.parameters():
                param.requires_grad_(True)
        self._log_global_param_delta(
            before=before_params,
            class_id=class_id,
            trigger=trigger,
            level=level,
            round_idx=round_idx,
        )
        epochs_run = len(hist.get("epoch_loss", []))
        self._log_replay_delta(
            before=before_replay,
            class_id=class_id,
            level=-1 if level is None else level,
            round_idx=round_idx or 0,
            phase=f"coupling_{trigger}",
            new_samples=self._phase_new_sample_count(loader, epochs_run),
        )
        if self.logger:
            prefix = (
                f"diagnostics/coupling/class_{class_id}/{trigger}/"
                f"level_{level if level is not None else 'none'}/"
                f"round_{round_idx if round_idx is not None else 'none'}"
            )
            final_loss = hist.get("epoch_loss", [None])[-1]
            metrics = {
                f"{prefix}/epochs": float(epochs_run),
                f"{prefix}/lr": float(lr),
                f"{prefix}/scope_{self.global_coupling_cfg.get('scope')}": 1.0,
            }
            if final_loss is not None:
                metrics[f"{prefix}/final_loss"] = float(final_loss)
            self.logger.log_metrics(metrics, step=self._class_count)

    @staticmethod
    def _phase_new_sample_count(loader: DataLoader, epochs_run: int) -> int:
        sampler = getattr(loader, "sampler", None)
        try:
            n_samples = len(sampler) if sampler is not None else len(loader.dataset)
        except Exception:
            try:
                n_samples = len(loader) * int(getattr(loader, "batch_size", 1) or 1)
            except Exception:
                n_samples = 0
        return int(n_samples) * int(max(epochs_run, 0))

    def _get_recon_errors(self, loader: DataLoader, level: int) -> Tensor:
        """
        Compute reconstruction errors at specified encoder level for all samples in loader.
        Returns a CPU tensor to avoid building gigantic autograd graphs on the GPU.
        """
        device = self._model_device()
        errors: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                x_hat = self.ae.forward_partial(x, level)
                err = self.ae.reconstruction_error(x_hat, x).detach().cpu()
                errors.append(err)
        return torch.cat(errors) if errors else torch.empty(0)

    def _should_log_outlier_criterion_diagnostics(self, level: int) -> bool:
        cfg = self.outlier_criterion_diagnostics
        if not cfg.get("enabled", False) or not self.logger:
            return False
        levels = cfg.get("levels", set())
        return not levels or int(level) in levels

    def _log_outlier_criterion_diagnostics(
        self,
        *,
        loader: DataLoader,
        level: int,
        class_id: Any,
        iteration: int,
        pixel_errors: Tensor,
        pixel_mask: Tensor,
    ) -> None:
        if not self._should_log_outlier_criterion_diagnostics(level):
            return
        local_errors: list[Tensor] = []
        device = self._model_device()
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                local_recon, local_target = self.ae.forward_level_ae(
                    x, level, ret_target=True
                )
                err = self.ae.reconstruction_error(local_recon, local_target)
                local_errors.append(err.detach().cpu())
        if not local_errors:
            return
        local = torch.cat(local_errors)
        pixel = pixel_errors.detach().cpu()
        pixel_mask = pixel_mask.detach().cpu().bool()
        if local.numel() != pixel.numel():
            return

        n_pixel = int(pixel_mask.sum().item())
        local_topk_mask = torch.zeros_like(pixel_mask)
        if n_pixel > 0:
            _, local_idx = torch.topk(local, k=min(n_pixel, local.numel()))
            local_topk_mask[local_idx] = True

        overlap = int((pixel_mask & local_topk_mask).sum().item())
        union = int((pixel_mask | local_topk_mask).sum().item())
        prefix = (
            f"diagnostics/outlier_criterion/class_{class_id}/"
            f"level_{level}/iteration_{iteration}"
        )
        metrics = {
            f"{prefix}/pixel_mean": float(pixel.mean().item()),
            f"{prefix}/pixel_std": float(pixel.std(unbiased=False).item()),
            f"{prefix}/pixel_threshold": float(self.thresholds[level]),
            f"{prefix}/pixel_outlier_fraction": float(pixel_mask.float().mean().item()),
            f"{prefix}/local_mean": float(local.mean().item()),
            f"{prefix}/local_std": float(local.std(unbiased=False).item()),
            f"{prefix}/local_topk_fraction": float(local_topk_mask.float().mean().item()),
            f"{prefix}/overlap_fraction_of_pixel": float(overlap / max(n_pixel, 1)),
            f"{prefix}/jaccard": float(overlap / max(union, 1)),
        }
        for percentile in self.outlier_criterion_diagnostics.get("percentiles", []):
            q = min(max(float(percentile), 0.0), 1.0)
            metrics[f"{prefix}/local_p{int(q * 1000):03d}"] = float(
                torch.quantile(local, q).item()
            )
            metrics[f"{prefix}/pixel_p{int(q * 1000):03d}"] = float(
                torch.quantile(pixel, q).item()
            )
        if local.numel() > 1 and float(local.std(unbiased=False).item()) > 0.0 and float(
            pixel.std(unbiased=False).item()
        ) > 0.0:
            corr = torch.corrcoef(torch.stack([pixel, local]))[0, 1]
            metrics[f"{prefix}/pixel_local_corr"] = float(corr.item())
        self.logger.log_metrics(metrics, step=self._class_count)

    def _get_outliers(
        self,
        loader: DataLoader,
        level: int,
        class_id: Any | None = None,
        iteration: int = 0,
    ):
        device = self._model_device()

        # 1) Collect all per-sample errors into a single flat Tensor
        errors = []
        for batch in loader:
            x, _ = batch
            x = x.to(device, non_blocking=True)
            err = self.ae.reconstruction_error(
                self.ae.forward_partial(x, level), x
            )  # shape: [batch_size]
            errors.append(err.detach().cpu())
        errors = torch.cat(errors)  # [N]
        stats_key = (str("unknown" if class_id is None else class_id), int(level), int(iteration))
        self._latest_outlier_stats[stats_key] = {
            "pixel_mean": float(errors.mean().item()) if errors.numel() else float("nan"),
            "pixel_std": float(errors.std(unbiased=False).item()) if errors.numel() else float("nan"),
            "pixel_max": float(errors.max().item()) if errors.numel() else float("nan"),
        }
        # print(f"[DEBUG] Errors[:10] = {errors[:10].tolist()}  mean={errors.mean().item():.4f}")

        # 2) Compare to the configured threshold, optionally calibrated from
        # the post-training error distribution for diagnostic ablations.
        threshold_info = self._effective_outlier_threshold(
            errors=errors, level=level, iteration=iteration
        )
        thr = float(threshold_info["threshold"])
        mask = errors > thr
        self._log_adaptive_outlier_threshold(
            class_id="unknown" if class_id is None else class_id,
            level=level,
            iteration=iteration,
            total_seen=int(errors.numel()),
            threshold_info=threshold_info,
        )
        self._log_outlier_criterion_diagnostics(
            loader=loader,
            level=level,
            class_id="unknown" if class_id is None else class_id,
            iteration=iteration,
            pixel_errors=errors,
            pixel_mask=mask,
        )
        # print(f"[DEBUG] threshold = {thr:.4f}, errors>thr mask[:10] = {mask[:10].tolist()}")
        n_outliers = int(mask.sum().item())
        # print(f"[DEBUG] n_outliers = {n_outliers} / {len(errors)}")

        # 3) Find the *real* dataset indices you iterated over
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "idxs"):
            all_indices = loader.sampler.idxs
        else:
            # if loader.dataset is a Subset, its .indices attr points into the full dataset
            all_indices = getattr(loader.dataset, "indices", list(range(len(loader.dataset))))
        # print(f"[DEBUG] first 10 all_indices = {all_indices[:10]}")

        # 4) Map mask→real indices
        outlier_real_idxs = [all_indices[i] for i, m in enumerate(mask) if m]
        # print(f"[DEBUG] first 10 outlier_real_idxs = {outlier_real_idxs[:10]}")

        # 5) Build subset & return loader
        # If the original loader.dataset is already a Subset, its `.indices`
        # refer to the underlying base dataset. We must create the new Subset
        # from that base dataset, not from the Subset itself, otherwise the
        # indices would be interpreted relative to the Subset and can go
        # out-of-range. For non-Subset datasets, use the dataset directly.
        base_ds = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
        subset = Subset(base_ds, outlier_real_idxs)
        outlier_loader = DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        total = int(errors.numel())
        return n_outliers, outlier_loader, total

    def _limit_loader(self, loader: DataLoader, n_samples: int) -> DataLoader:
        """
        Returns a DataLoader that yields only the first n_samples from the original loader.
        """
        total = min(n_samples, len(loader.dataset))
        limited_idxs = list(range(total))
        limited_subset = Subset(loader.dataset, limited_idxs)
        return DataLoader(
            limited_subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )

    def log_global_sizes(self):
        if self.logger:
            global_metrics = {
                f"global_level_{i}_size": sz for i, sz in enumerate(self.ae.hidden_sizes)
            }
            self.logger.log_metrics(global_metrics, step=self._class_count)

    def _build_epoch_logger(self, class_id: Any, level: int, round_idx: int):
        if not self.logger:
            return None

        metric_prefix = f"class_{class_id}"
        artifact_prefix = f"plasticity/class_{class_id}/level_{level}/round_{round_idx}"
        log_dict_fn = getattr(self.logger, "log_dict", None)
        log_metrics_fn = getattr(self.logger, "log_metrics", None)

        def _callback(epoch_idx: int, summary: dict):
            phase = summary.get("phase", "train")
            if callable(log_dict_fn):
                payload = {
                    "class_id": class_id,
                    "level": level,
                    "round": round_idx,
                    "phase": phase,
                    "epoch": epoch_idx + 1,
                    **summary,
                }
                log_dict_fn(payload, f"{artifact_prefix}/epoch_{epoch_idx + 1}.json")
            if "loss" in summary:
                key = (class_id, level, round_idx, str(phase))
                self._phase_loss_history.setdefault(key, []).append(float(summary["loss"]))
            if callable(log_metrics_fn) and "loss" in summary:
                loss = float(summary["loss"])
                metrics = {
                    f"{phase}/class_{class_id}/level_{level}/round_{round_idx}_loss": loss
                }
                log_metrics_fn(metrics, step=epoch_idx + 1)
            if callable(log_metrics_fn) and "eval_loss" in summary:
                eval_loss = float(summary["eval_loss"])
                metrics = {
                    f"{phase}/class_{class_id}/level_{level}/round_{round_idx}_eval_loss": eval_loss
                }
                log_metrics_fn(metrics, step=epoch_idx + 1)

        return _callback

    def _log_phase_loss_plot(self, class_id: Any, level: int, round_idx: int) -> None:
        if not self.logger:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        plast_key = (class_id, level, round_idx, "plasticity")
        stab_key = (class_id, level, round_idx, "stability")
        plast = self._phase_loss_history.get(plast_key, [])
        stab = self._phase_loss_history.get(stab_key, [])
        if not plast and not stab:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        if plast:
            ax.plot(range(1, len(plast) + 1), plast, label="plasticity")
        if stab:
            ax.plot(range(1, len(stab) + 1), stab, label="stability")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Class {class_id} Level {level} Round {round_idx}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        artifact = f"figures/phase_losses/class_{class_id}/level_{level}/round_{round_idx}.png"
        try:
            self.logger.log_figure(fig, artifact)
        finally:
            plt.close(fig)

    def set_recon_eval_batch(self, batch: torch.Tensor | None) -> None:
        self._recon_eval_batch = batch

    def learn_class(self, class_id: Any, loader: DataLoader) -> None:
        num_layers = len(self.ae.hidden_sizes)
        sizes_before = list(self.ae.hidden_sizes)
        self.history[class_id] = {"layer_errors": [[] for _ in range(num_layers)]}
        outlier_history: dict[int, list[dict]] = {lvl: [] for lvl in range(num_layers)}

        # ---- log "class learned" ----
        self._class_count += 1
        if self.logger:
            self.logger.log_metrics({"classes_learned": self._class_count}, step=self._class_count)

        device = self._model_device()
        n_old_classes = len(self.ir.available_classes()) if self.ir is not None else 0
        replay_sampler, replay_only = self._build_replay_sampler(
            device, n_old_classes=n_old_classes
        )

        pbar_levels = tqdm(range(num_layers), desc=f"[Class {class_id}] Layers", unit="lvl")
        for level in pbar_levels:
            added = 0
            self.ae._plastic_to_mature()
            n_plastic_neurons = 0
            iteration_idx = 0
            n_outliers, outliers_loader, total_seen = self._get_outliers(
                loader, level, class_id=class_id, iteration=iteration_idx
            )
            fraction = n_outliers / max(total_seen, 1)
            self._log_outlier_metrics(
                class_id=class_id,
                level=level,
                iteration=iteration_idx,
                n_outliers=n_outliers,
                total_seen=total_seen,
            )
            outlier_history[level].append(
                {
                    "iteration": iteration_idx,
                    "n_outliers": n_outliers,
                    "total_seen": total_seen,
                    "fraction": fraction,
                }
            )

            if self.logger:
                self.logger.log_metrics(
                    {f"class_{class_id}/growth_level_{level}": self.ae.hidden_sizes[level]},
                    step=added,
                )
            max_new_nodes = int(self.max_nodes[level])
            pbar_growth = tqdm(
                range(max_new_nodes), desc=f"  Level {level} Growth", unit="rnd", leave=False
            )

            step_plasticety = 0
            step_stability = 0

            for _ in pbar_growth:
                shape_info = self._shape_pressure(level)
                base_outliers_allowed = self._max_outliers_allowed(
                    total_seen, level=level
                )
                max_outliers_allowed = self._effective_max_outliers_allowed(
                    level=level, total_seen=total_seen, shape_info=shape_info
                )
                quality_info = self._quality_growth_gate(
                    class_id=class_id, level=level, iteration=iteration_idx
                )
                self._log_growth_gate(
                    class_id=class_id,
                    level=level,
                    round_idx=added + 1,
                    n_outliers=n_outliers,
                    total_seen=total_seen,
                    base_allowed=base_outliers_allowed,
                    effective_allowed=max_outliers_allowed,
                    shape_info=shape_info,
                    quality_info=quality_info,
                )
                outlier_gate_passes = n_outliers > max_outliers_allowed
                quality_gate_passes = bool(quality_info["quality_passes"])
                if not (
                    outlier_gate_passes
                    and quality_gate_passes
                    and n_plastic_neurons < max_new_nodes
                ):
                    break

                nodes_existing = self.ae.encoder[2 * level].n_out_features
                request = self._growth_request(
                    level=level,
                    n_outliers=n_outliers,
                    nodes_existing=nodes_existing,
                    n_plastic_neurons=n_plastic_neurons,
                    shape_info=shape_info,
                )
                num_new = int(request["actual_new_nodes"])
                self._log_growth_request(
                    class_id=class_id,
                    level=level,
                    round_idx=added + 1,
                    n_outliers=n_outliers,
                    total_seen=total_seen,
                    request=request,
                )
                if num_new <= 0:
                    break

                self.ae.add_new_nodes(level, num_new)
                n_plastic_neurons += num_new
                if self.logger:
                    self.logger.log_metrics(
                        {f"class_{class_id}_level_{level}_n_plastic_neurons": n_plastic_neurons},
                        step=added,
                    )
                    current_sizes = {
                        f"class_{class_id}_level_{lvl}_cumulative_size": sz
                        for lvl, sz in enumerate(self.ae.hidden_sizes)
                    }
                    self.logger.log_metrics(current_sizes, step=added)
                if self.logger:
                    self.logger.log_metrics(
                        {f"class_{class_id}/growth_level_{level}": self.ae.hidden_sizes[level]},
                        step=added,
                    )

                last_loss = 1
                round_idx = added + 1
                epoch_logger = self._build_epoch_logger(class_id, level, round_idx)
                phase_es_cfg = self._build_phase_early_stop_cfg(level, phase="plasticity")
                param_before = (
                    self._snapshot_level_params(level) if self.logger else None
                )
                replay_before = self._snapshot_replay_counters()

                hist = self.ae.plasticity_phase(
                    loader=outliers_loader,
                    level=level,
                    epochs=self.plasticity_epochs,
                    lr=self.base_lr,
                    early_stop_cfg=phase_es_cfg,
                    forward_fn=self._phase_forward_fn(level, "plasticity"),
                    epoch_logger=epoch_logger,
                )
                if param_before is not None:
                    self._log_param_delta(
                        before=param_before,
                        class_id=class_id,
                        level=level,
                        round_idx=round_idx,
                        phase="plasticity",
                    )
                self._log_replay_delta(
                    before=replay_before,
                    class_id=class_id,
                    level=level,
                    round_idx=round_idx,
                    phase="plasticity",
                    new_samples=self._phase_new_sample_count(
                        outliers_loader, len(hist.get("epoch_loss", []))
                    ),
                )

                mean_loss = hist["epoch_loss"][-1]
                delta_loss = last_loss - mean_loss if last_loss is not None else float("inf")

                if self.logger:
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_plasticity": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_plasticity": delta_loss,
                        },
                        step_plasticety,
                    )
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_plasticity_iter": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_plasticity_iter": delta_loss,
                        },
                        step=self._class_count,
                    )
                step_plasticety += 1
                last_loss = mean_loss

                last_loss = 1
                phase_es_cfg = self._build_phase_early_stop_cfg(level, phase="stability")
                param_before = (
                    self._snapshot_level_params(level) if self.logger else None
                )
                replay_before = self._snapshot_replay_counters()
                stability_kwargs = {
                    "loader": loader,
                    "level": level,
                    "lr": self.base_lr,
                    "epochs": self.stability_epochs,
                    "old_x": replay_sampler,
                    "replay_only": False,
                    "eval_batch": self._recon_eval_batch,
                    "early_stop_on_eval": self._recon_eval_batch is not None,
                    "early_stop_cfg": phase_es_cfg,
                    "forward_fn": self._phase_forward_fn(level, "stability"),
                    "epoch_logger": epoch_logger,
                    "replay_loss_weight": self.stability_replay_loss_weight,
                }
                hist = self._scheduled_stability_phase(**stability_kwargs)
                if param_before is not None:
                    self._log_param_delta(
                        before=param_before,
                        class_id=class_id,
                        level=level,
                        round_idx=round_idx,
                        phase="stability",
                    )
                self._log_replay_delta(
                    before=replay_before,
                    class_id=class_id,
                    level=level,
                    round_idx=round_idx,
                    phase="stability",
                    new_samples=self._phase_new_sample_count(
                        loader, int(hist.get("_new_epochs", len(hist.get("epoch_loss", []))))
                    ),
                )

                self._log_phase_loss_plot(class_id, level, round_idx)

                mean_loss = hist["epoch_loss"][-1]
                delta_loss = last_loss - mean_loss if last_loss is not None else float("inf")

                if self.logger:
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_stability": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_stability": delta_loss,
                        },
                        step_stability,
                    )
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_stability_iter": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_stability_iter": delta_loss,
                        },
                        step=self._class_count,
                    )
                step_stability += 1
                last_loss = mean_loss

                self._run_global_coupling(
                    class_id=class_id,
                    loader=loader,
                    replay_sampler=replay_sampler,
                    replay_only=replay_only,
                    trigger="after_growth_round",
                    level=level,
                    round_idx=round_idx,
                )

                errs = self._get_recon_errors(loader, level)
                self.history[class_id]["layer_errors"][level].append(errs.clone())
                if self.logger:
                    self.logger.log_metrics(
                        {f"class_{class_id}_level_{level}_avg_loss": errs.mean().item()},
                        step=added,
                    )
                    self.logger.log_metrics(
                        {f"class_{class_id}_level_{level}_avg_loss_iter": errs.mean().item()},
                        step=self._class_count,
                    )

                added += 1
                iteration_idx += 1
                n_outliers, outliers_loader, total_seen = self._get_outliers(
                    loader, level, class_id=class_id, iteration=iteration_idx
                )
                self._log_outlier_metrics(
                    class_id=class_id,
                    level=level,
                    iteration=iteration_idx,
                    n_outliers=n_outliers,
                    total_seen=total_seen,
                )
                fraction = n_outliers / max(total_seen, 1)
                outlier_history[level].append(
                    {
                        "iteration": iteration_idx,
                        "n_outliers": n_outliers,
                        "total_seen": total_seen,
                        "fraction": fraction,
                    }
                )
                if self.logger:
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_n_outliers": n_outliers,
                            f"class_{class_id}_level_{level}_outlier_fraction": fraction,
                        },
                        step=added,
                    )

            pbar_growth.close()

            # Next-layer plasticity & stability
            if level + 1 < num_layers and step_plasticety > 0:
                phase_es_cfg = self._build_phase_early_stop_cfg(level + 1, phase="plasticity")
                self.ae.plasticity_phase(
                    loader,
                    level + 1,
                    epochs=self.next_layer_epochs,
                    lr=self.base_lr * self.next_layer_lr_ratio,
                    early_stop_cfg=phase_es_cfg,
                    forward_fn=self._phase_forward_fn(level + 1, "plasticity"),
                    train_mature_encoder=True,
                )
                phase_es_cfg = self._build_phase_early_stop_cfg(level + 1, phase="stability")
                self._call_stability_phase(
                    loader=loader,
                    level=level + 1,
                    lr=self.base_lr * self.next_layer_lr_ratio,
                    epochs=self.stability_epochs,
                    old_x=replay_sampler,
                    replay_only=False,
                    eval_batch=self._recon_eval_batch,
                    early_stop_on_eval=self._recon_eval_batch is not None,
                    early_stop_cfg=phase_es_cfg,
                    forward_fn=self._phase_forward_fn(level + 1, "stability"),
                    replay_loss_weight=self.stability_replay_loss_weight,
                )

            self._run_global_coupling(
                class_id=class_id,
                loader=loader,
                replay_sampler=replay_sampler,
                replay_only=replay_only,
                trigger="after_level",
                level=level,
            )

            if self.logger:
                size = self.ae.hidden_sizes[level]
                self.logger.log_metrics(
                    {f"class_{class_id}_level_{level}_size": size}, step=self._class_count
                )

            pbar_levels.update(1)
        pbar_levels.close()
        self._run_global_coupling(
            class_id=class_id,
            loader=loader,
            replay_sampler=replay_sampler,
            replay_only=replay_only,
            trigger="after_class",
        )
        # Fit IR stats for the incoming class

        if self.logger and outlier_history:
            lines = [f"Outlier progression for class {class_id}"]
            for level_idx in range(num_layers):
                entries = outlier_history.get(level_idx) or []
                if not entries:
                    continue
                lines.append(f"Level {level_idx}:")
                for entry in entries:
                    iteration = entry["iteration"]
                    n_out = entry["n_outliers"]
                    total = entry["total_seen"]
                    frac = entry["fraction"]
                    lines.append(
                        f"  iter {iteration:02d}: {n_out} / {total} ({frac:.4f})"
                    )
            if len(lines) == 1:
                lines.append("No outlier measurements recorded.")
            text = "\n".join(lines)
            self.logger.log_text(text, f"neurogenesis/class_{class_id}_outliers.txt")
            final_metrics: dict[str, float] = {}
            for level_idx in range(num_layers):
                entries = outlier_history.get(level_idx) or []
                if not entries:
                    continue
                final = entries[-1]
                shape_info = self._shape_pressure(level_idx)
                allowed = self._max_outliers_allowed(
                    int(final["total_seen"]), level=level_idx
                )
                effective_allowed = self._effective_max_outliers_allowed(
                    level=level_idx,
                    total_seen=int(final["total_seen"]),
                    shape_info=shape_info,
                )
                growth = int(self.ae.hidden_sizes[level_idx] - sizes_before[level_idx])
                final_metrics[
                    f"diagnostics/outliers/class_{class_id}/level_{level_idx}_final_count"
                ] = float(final["n_outliers"])
                final_metrics[
                    f"diagnostics/outliers/class_{class_id}/level_{level_idx}_final_fraction"
                ] = float(final["fraction"])
                final_metrics[
                    f"diagnostics/outliers/class_{class_id}/level_{level_idx}_allowed_count"
                ] = float(allowed)
                final_metrics[
                    f"diagnostics/outliers/class_{class_id}/level_{level_idx}_effective_allowed_count"
                ] = float(effective_allowed)
                final_metrics[
                    f"diagnostics/outliers/class_{class_id}/level_{level_idx}_below_threshold"
                ] = 1.0 if int(final["n_outliers"]) <= effective_allowed else 0.0
                final_metrics[
                    f"diagnostics/growth/class_{class_id}/level_{level_idx}_hit_cap"
                ] = 1.0 if growth >= int(self.max_nodes[level_idx]) else 0.0
            if final_metrics:
                self.logger.log_metrics(final_metrics, step=self._class_count)

        loader = tqdm(
            loader,
            desc=f"[Class {class_id}] Fitting IR",
            unit="batch",
            leave=False,
        )
        if self.ir is not None:
            self.ir.fit(loader)
        self.log_global_sizes()

        # Refresh replay statistics with the now-updated encoder
        if self.ir is not None:
            self.ir.fit(loader)

        if self.logger and sizes_before:
            summary_metrics = {}
            for idx, (before, after) in enumerate(zip(sizes_before, self.ae.hidden_sizes)):
                summary_metrics[f"summary/layer_{idx}_growth_total"] = after - before
                summary_metrics[f"summary/layer_{idx}_cumulative_size"] = after
            self.logger.log_metrics(summary_metrics, step=self._class_count)

    def test_all_levels(self, loader: DataLoader) -> List[float]:
        """
        Evaluate the autoencoder at each encoder depth and return the mean reconstruction loss per level.
        """
        mean_losses: List[float] = []
        max_losses: List[float] = []
        std_losses: List[float] = []

        # Ensure model is in evaluation mode
        self.ae.eval()
        with torch.no_grad():
            for level in range(len(self.ae.hidden_sizes)):
                # Compute reconstruction errors at this level
                errors = self._get_recon_errors(loader, level)
                # Mean loss for the level
                mean_loss = errors.mean().item()
                max_loss = errors.max().item()
                std_loss = errors.std().item()
                mean_losses.append(mean_loss)
                max_losses.append(max_loss)
                std_losses.append(std_loss)

                # Log metric if logger is available
                if self.logger:
                    self.logger.log_metrics({"test_mean_loss": mean_loss}, level)
                    self.logger.log_metrics({"test_max_loss": max_loss}, level)
                    self.logger.log_metrics({"test_std_loss": std_loss}, level)

        # Restore training mode
        self.ae.train()
        return mean_losses, max_losses, std_losses
