from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitCheckResult:
    users_checked: int
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_if_failed(self) -> None:
        if self.violations:
            preview = "; ".join(self.violations[:5])
            raise ValueError(f"Chronological split check failed: {preview}")


def check_chronological_splits(
    train,
    val,
    test,
    user_col: str = "user_idx",
    time_col: str = "timestamp",
    pos_col: str = "pos",
) -> SplitCheckResult:
    violations: list[str] = []
    users = (
        set(train[user_col].unique()) | set(val[user_col].unique()) | set(test[user_col].unique())
    )

    def order_value(frame):
        cols = [time_col]
        if pos_col in frame.columns:
            cols.append(pos_col)
        return frame[cols].max().to_list(), frame[cols].min().to_list()

    for user in sorted(int(x) for x in users):
        tr = train[train[user_col] == user]
        va = val[val[user_col] == user]
        te = test[test[user_col] == user]
        if tr.empty or va.empty or te.empty:
            violations.append(f"user {user}: missing train/val/test rows")
            continue
        tr_max, _ = order_value(tr)
        _, va_min = order_value(va)
        va_max, _ = order_value(va)
        _, te_min = order_value(te)
        if tuple(tr_max) > tuple(va_min):
            violations.append(f"user {user}: train occurs after validation")
        if tuple(va_max) > tuple(te_min):
            violations.append(f"user {user}: validation occurs after test")

    return SplitCheckResult(users_checked=len(users), violations=violations)
