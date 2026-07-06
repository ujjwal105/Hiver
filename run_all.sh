#!/usr/bin/env bash
# End-to-end: build data -> generate replies -> score them -> validate the metric.
# Uses whatever provider is available (anthropic if ANTHROPIC_API_KEY is set, else mock).
set -e
PROV="${1:-}"          # optional: pass "mock" or "anthropic" to force
ARG=""; [ -n "$PROV" ] && ARG="--provider $PROV"

python3 -m src.dataset --augment
python3 -m src.generate --split test --out outputs/generated.jsonl $ARG
python3 -m src.evaluate --in outputs/generated.jsonl --out outputs/scored.jsonl $ARG
python3 -m src.validate_metric --out outputs/metric_validation.json $ARG
echo "Done. See outputs/ (generated.jsonl, scored.jsonl, report.json, metric_validation.json)"
