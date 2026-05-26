# Phase 1 Ordinal Scorer Report

## Summary

| Channel | Status | Epochs | Ord MAE | Exp MAE | Within-1 | F1@0.5 | Best Sweep F1 | Flag |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| normal_map | ok | 30 | 1.2235 | 0.6391 | 0.5711 | 0.8615 | 0.8665 | good |
| roughness | ok | 30 | 1.5321 | 0.6904 | 0.4584 | 0.6813 | 0.7995 | good |
| metallic | ok | 23 | 1.1190 | 0.6843 | 0.6877 | 0.8038 | 0.8119 | good |
| base_color | ok | 10 | 1.8879 | 0.6185 | 0.2275 | 0.3185 | 0.3361 | good |

## Best Epochs

| Channel | Ord MAE | Exp MAE | Within-1 | F1@0.5 | Best Sweep F1 |
|---|---:|---:|---:|---:|---:|
| normal_map | 20 | 23 | 23 | 21 | 21 |
| roughness | 24 | 24 | 28 | 5 | 30 |
| metallic | 16 | 16 | 16 | 16 | 14 |
| base_color | 1 | 1 | 3 | 1 | 9 |

## Notes

- For continuous quality scoring, prefer probability expected score over argmax class.
- `expected_mae` is the most relevant quick metric for continuous score calibration.
- `binary_best_f1` is kept only to compare against the old pass/fail gate.
- Channels flagged `weak` should still be usable for embedding features, but not trusted as standalone score heads.

