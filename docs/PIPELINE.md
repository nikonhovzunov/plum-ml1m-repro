# Pipeline

```mermaid
flowchart LR
    A["MovieLens-1M ratings/movies/users"] --> B["Chronological leave-one-out split"]
    A --> C["Movie metadata + plot overviews"]
    C --> D["Text embeddings"]
    B --> E["Train-only behavior pairs"]
    D --> F["RQ-VAE Semantic IDs"]
    E --> F
    F --> G["CPT corpus"]
    C --> G
    G --> H["CPT LoRA/QLoRA adapter / merged checkpoint"]
    H --> I["SFT LoRA/QLoRA adapter"]
    I --> J["Trie-constrained SID decoding"]
    J --> K["SID-to-item mapping + seen filtering"]
    K --> L["Ranking metrics"]
```

## CPT corpus

The active Qwen CPT series uses a pre-built curriculum corpus:

- `15` synthetic corpus epochs.
- One Trainer epoch over the generated corpus.
- `50%` behavior windows and `50%` metadata/content examples.
- Behavior windows are built only from the train split.
- Metadata examples cover all catalog items each synthetic epoch.
- Description examples are sharded across synthetic epochs.

## SFT task

SFT is next watched item prediction. The model sees a fixed user history window
and generates the target item SID. Loss is applied to target tokens only.

The current main Qwen3 run uses a window of `16` visible history items and
filters recommendations against the user's full prior history during
validation/test evaluation.

## Decoding

Decoding is constrained by a trie over valid item SID token sequences. Generated SID sequences are mapped back to original item indices before metric computation.
