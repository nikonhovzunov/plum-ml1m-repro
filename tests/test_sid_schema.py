import numpy as np
import pytest

from plum_ml1m.protocol import ACTIVE_SID_PROTOCOL
from plum_ml1m.sid import (
    ItemSIDMapping,
    SIDTokenError,
    format_sid,
    format_sid_token,
    parse_sid,
    parse_sid_token,
    try_parse_sid,
)


def test_sid_schema_roundtrip():
    sid = (511, 255, 127, 63)
    tokens = format_sid(sid)
    assert tokens == [
        "<sid_0_511>",
        "<sid_1_255>",
        "<sid_2_127>",
        "<sid_3_63>",
    ]
    assert parse_sid(tokens) == sid
    assert ACTIVE_SID_PROTOCOL.n_levels == 4


def test_malformed_sid_tokens():
    with pytest.raises(SIDTokenError):
        parse_sid_token("sid_0_1")
    with pytest.raises(SIDTokenError):
        parse_sid(["<sid_0_1>", "<sid_0_2>", "<sid_2_3>", "<sid_3_4>"])
    assert try_parse_sid(["<sid_0_1>", "<sid_1_2>"]) is None


def test_sid_validation_rejects_wrong_level_count_and_code_range():
    with pytest.raises(ValueError):
        format_sid((1, 2, 3))
    with pytest.raises(ValueError):
        format_sid((0, 0, 0, 128))


def test_item_sid_mapping_collision_expand_and_seen_filtering():
    sids = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )
    mapping = ItemSIDMapping.from_sids(sids)
    assert mapping.n_collision_buckets == 1
    assert mapping.uniqueness == pytest.approx(2 / 3)
    assert mapping.resolve_sid((1, 2, 3, 4), policy="expand") == [0, 1]
    assert mapping.resolve_sid((1, 2, 3, 4), policy="representative") == [0]
    assert mapping.sid_candidates_to_items([(1, 2, 3, 4), (5, 6, 7, 8)], k=10, seen_items={0}) == [
        1,
        2,
    ]


def test_format_single_sid_token():
    assert format_sid_token(2, 17) == "<sid_2_17>"
    assert parse_sid_token("<sid_2_17>") == (2, 17)
