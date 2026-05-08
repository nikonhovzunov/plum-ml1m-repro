from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SIDProtocol:
    """Semantic ID protocol definition used by the active SID-v2 experiments."""

    name: str
    codebook_sizes: tuple[int, ...]
    sid_token_prefix: str = "sid"
    duplicate_token_prefix: str = "dup"
    collision_policy: str = "expand"

    @property
    def n_levels(self) -> int:
        return len(self.codebook_sizes)

    def validate_sid(self, sid: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        values = tuple(int(x) for x in sid)
        if len(values) != self.n_levels:
            raise ValueError(f"{self.name} expects {self.n_levels} SID levels, got {len(values)}")
        for level, code in enumerate(values):
            size = int(self.codebook_sizes[level])
            if code < 0 or code >= size:
                raise ValueError(
                    f"SID code out of range at level {level}: {code} not in [0, {size})"
                )
        return values


ACTIVE_SID_PROTOCOL = SIDProtocol(
    name="sid-v2",
    codebook_sizes=(1024, 512, 256, 128),
    collision_policy="expand",
)
