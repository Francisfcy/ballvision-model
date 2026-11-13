from __future__ import annotations

import math
from collections import Counter, defaultdict

from .schemas import IdentityCandidate, PlayerProfile, TrackIdentity, TrackObservation


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


class IdentityResolver:
    """Resolve track identities without requiring jersey numbers.

    Jersey numbers add evidence when visible. They do not gate identity resolution.
    A real detector supplies appearance embeddings and optional team hints.
    """

    def __init__(self, roster: list[PlayerProfile]) -> None:
        self.roster = roster

    def resolve(
        self,
        observations: list[TrackObservation],
        *,
        manual_bindings: dict[str, str] | None = None,
    ) -> list[TrackIdentity]:
        manual_bindings = manual_bindings or {}
        grouped: dict[str, list[TrackObservation]] = defaultdict(list)
        for observation in observations:
            grouped[observation.track_id].append(observation)

        resolved: list[TrackIdentity] = []
        for track_id, track_observations in grouped.items():
            if track_id in manual_bindings:
                resolved.append(
                    TrackIdentity(
                        track_id=track_id,
                        candidates=[IdentityCandidate(player_id=manual_bindings[track_id], score=1.0, reasons=["manual_binding"])],
                        status="confirmed",
                        confirmed_player_id=manual_bindings[track_id],
                    )
                )
                continue

            candidate_scores: Counter[str] = Counter()
            reasons: dict[str, set[str]] = defaultdict(set)
            for profile in self.roster:
                score_total = 0.0
                score_count = 0
                for observation in track_observations:
                    if observation.team_hint and observation.team_hint == profile.team_id:
                        score_total += 0.3
                        reasons[profile.player_id].add("team_hint")
                    if observation.appearance_embedding and profile.player_id in _EMBEDDINGS:
                        score_total += 0.6 * _cosine(observation.appearance_embedding, _EMBEDDINGS[profile.player_id])
                        score_count += 1
                        reasons[profile.player_id].add("appearance")
                    if observation.jersey_number and profile.jersey_number == observation.jersey_number:
                        score_total += 0.5
                        reasons[profile.player_id].add("optional_jersey_number")
                if score_total:
                    candidate_scores[profile.player_id] = score_total / max(1, len(track_observations) + score_count)

            candidates = [
                IdentityCandidate(player_id=player_id, score=min(1.0, score), reasons=sorted(reasons[player_id]))
                for player_id, score in candidate_scores.most_common()
            ]
            status = "candidate" if candidates else "unresolved"
            resolved.append(TrackIdentity(track_id=track_id, candidates=candidates[:3], status=status))
        return resolved


# The baseline keeps an embedding registry interface. A trained ReID service will populate it.
_EMBEDDINGS: dict[str, list[float]] = {}
