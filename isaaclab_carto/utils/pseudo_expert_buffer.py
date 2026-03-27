from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json
import os
import random

import torch


@dataclass
class EpisodeRecord:
    """
    Compact episode-level record for pseudo-expert selection.
    """
    preset_name: str
    terrain_id: str
    success: bool
    episode_length: int

    beta: List[float]
    command: List[float]

    return_total: float
    return_velocity: float
    return_slip: float
    return_energy: float

    mean_velocity: float
    mean_slip: float
    mean_energy: float

    policy_step: int
    timestamp: str

    # Optional selector supervision inputs
    latent: Optional[List[float]] = None
    context: Optional[List[float]] = None
    proprio_summary: Optional[List[float]] = None

    # Optional ranking / filtering metadata
    score: Optional[float] = None
    selected_as_pseudo_expert: bool = False

    ended_by_termination: bool = False
    ended_by_truncation: bool = False


class PseudoExpertBuffer:
    """
    Stores episode-level summaries and selects pseudo-expert samples.

    This is not a replay buffer for RL updates.
    It is a lightweight dataset builder for objective-selector supervision.
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.records: List[EpisodeRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: EpisodeRecord) -> None:
        if len(self.records) >= self.capacity:
            self.records.pop(0)
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records]

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PseudoExpertBuffer":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        buffer = cls(capacity=max(len(raw), 1))
        for item in raw:
            buffer.add(EpisodeRecord(**item))
        return buffer

    def filter_success(self) -> List[EpisodeRecord]:
        return [r for r in self.records if r.success]

    def filter_by_terrain(self, terrain_id: str) -> List[EpisodeRecord]:
        return [r for r in self.records if r.terrain_id == terrain_id]

    def filter_by_preset(self, preset_name: str) -> List[EpisodeRecord]:
        return [r for r in self.records if r.preset_name == preset_name]

    def compute_score(
        self,
        record: EpisodeRecord,
        w_success: float = 2.0,
        w_velocity: float = 1.0,
        w_slip: float = 1.0,
        w_energy: float = 1.0,
    ) -> float:
        """
        Higher is better.

        Assumes:
        - mean_velocity: higher is better
        - mean_slip: already reward-shaped (less slip => larger / less negative is better)
        - mean_energy: already reward-shaped (less torque => larger / less negative is better)
        """
        success_bonus = w_success if record.success else 0.0
        score = (
            success_bonus
            + w_velocity * record.mean_velocity
            + w_slip * record.mean_slip
            + w_energy * record.mean_energy
        )
        return float(score)

    def annotate_scores(self, **score_kwargs) -> None:
        for i, record in enumerate(self.records):
            self.records[i].score = self.compute_score(record, **score_kwargs)

    def select_top_k(
        self,
        k: int,
        ensure_success: bool = True,
        diversify_by_terrain: bool = False,
        min_episode_length: int = 50,
        min_mean_velocity: float = -1e9,
        min_mean_slip: float = -1e9,
        min_mean_energy: float = -1e9,
    ) -> List[EpisodeRecord]:
        candidates = self.filter_candidates(
            min_episode_length=min_episode_length,
            min_mean_velocity=min_mean_velocity,
            min_mean_slip=min_mean_slip,
            min_mean_energy=min_mean_energy,
            require_success=ensure_success,
        )

        if not candidates:
            return []

        for r in candidates:
            if r.score is None:
                r.score = self.compute_score(r)

        score_key = lambda x: float(x.score if x.score is not None else float("-inf"))

        if not diversify_by_terrain:
            ranked = sorted(candidates, key=score_key, reverse=True)
            selected = ranked[:k]
        else:
            grouped: Dict[str, List[EpisodeRecord]] = {}
            for r in candidates:
                grouped.setdefault(r.terrain_id, []).append(r)

            selected = []
            for _, group in grouped.items():
                group_sorted = sorted(group, key=score_key, reverse=True)
                selected.append(group_sorted[0])

            selected = sorted(selected, key=score_key, reverse=True)[:k]

        selected_ids = set(id(r) for r in selected)
        for r in self.records:
            r.selected_as_pseudo_expert = id(r) in selected_ids

        return selected
    
    def select_top_k_per_preset(
        self,
        k_per_preset: int,
        ensure_success: bool = True,
        min_episode_length: int = 50,
        min_mean_velocity: float = -1e9,
        min_mean_slip: float = -1e9,
        min_mean_energy: float = -1e9,
    ) -> List[EpisodeRecord]:
        candidates = self.filter_candidates(
            min_episode_length=min_episode_length,
            min_mean_velocity=min_mean_velocity,
            min_mean_slip=min_mean_slip,
            min_mean_energy=min_mean_energy,
            require_success=ensure_success,
        )

        if not candidates:
            return []

        for r in candidates:
            if r.score is None:
                r.score = self.compute_score(r)

        score_key = lambda x: float(x.score if x.score is not None else float("-inf"))

        grouped: Dict[str, List[EpisodeRecord]] = {}
        for r in candidates:
            grouped.setdefault(r.preset_name, []).append(r)

        selected: List[EpisodeRecord] = []
        for _, group in grouped.items():
            group_sorted = sorted(group, key=score_key, reverse=True)
            selected.extend(group_sorted[:k_per_preset])

        selected_ids = set(id(r) for r in selected)
        for r in self.records:
            r.selected_as_pseudo_expert = id(r) in selected_ids

        return selected

    def sample(self, batch_size: int) -> List[EpisodeRecord]:
        if len(self.records) == 0:
            return []
        batch_size = min(batch_size, len(self.records))
        return random.sample(self.records, batch_size)

    def build_selector_dataset(
        self,
        only_selected: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Converts records to training samples for the objective selector.

        Each sample contains:
        - input: latent/context + command
        - target: beta
        """
        if only_selected:
            source = [r for r in self.records if r.selected_as_pseudo_expert]
        else:
            source = list(self.records)

        dataset = []
        for r in source:
            if r.command is None or r.beta is None:
                continue

            sample = {
                "terrain_id": r.terrain_id,
                "preset_name": r.preset_name,
                "command": r.command,
                "beta": r.beta,
                "success": r.success,
                "score": r.score,
                "latent": r.latent,
                "context": r.context,
                "proprio_summary": r.proprio_summary,
            }
            dataset.append(sample)

        return dataset
    def filter_candidates(
        self,
        min_episode_length: int = 50,
        min_mean_velocity: float = -1e9,
        min_mean_slip: float = -1e9,
        min_mean_energy: float = -1e9,
        require_success: bool = True,
    ) -> List[EpisodeRecord]:
        candidates = []

        for r in self.records:
            if require_success and not r.success:
                continue
            if r.episode_length < min_episode_length:
                continue
            if r.mean_velocity < min_mean_velocity:
                continue
            if r.mean_slip < min_mean_slip:
                continue
            if r.mean_energy < min_mean_energy:
                continue
            candidates.append(r)

        return candidates