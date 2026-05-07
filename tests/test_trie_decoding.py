import numpy as np

from plum_ml1m.decoding import TrieConstrainedDecoder, filter_seen_items
from plum_ml1m.sid import ItemSIDMapping
from plum_ml1m.trie import TokenTrie


def test_trie_valid_and_invalid_sequences():
    trie = TokenTrie.from_sequences([[10, 20], [10, 21]], eos_id=99)
    assert trie.contains([10, 20])
    assert trie.contains([10, 21])
    assert not trie.contains([10, 22])
    assert trie.next_tokens([]) == [10]
    assert trie.next_tokens([10]) == [20, 21]
    assert trie.next_tokens([10, 20]) == [99]
    assert trie.next_tokens([123]) == [99]


def test_constrained_decoder_maps_token_sequences_to_items_with_collisions():
    sids = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 8, 7, 6]])
    mapping = ItemSIDMapping.from_sids(sids)
    trie = TokenTrie.from_sequences([[10, 20, 30, 40], [11, 21, 31, 41]], eos_id=99)
    decoder = TrieConstrainedDecoder(
        trie=trie,
        token_ids_to_sid={
            (10, 20, 30, 40): (1, 2, 3, 4),
            (11, 21, 31, 41): (9, 8, 7, 6),
        },
        sid_mapping=mapping,
    )

    recs = decoder.recommend_from_token_sequences(
        [[10, 20, 30, 40, 99], [11, 21, 31, 41, 99]],
        k=10,
        seen_items={0},
    )
    assert recs == [1, 2]


def test_seen_item_filtering_and_duplicate_predictions():
    assert filter_seen_items([1, 1, 2, 3, 2], seen_items={2}) == [1, 3]
