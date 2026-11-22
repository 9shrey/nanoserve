#!/bin/bash
# Full benchmark suite; every table in the README comes from here. ~45 min on a GTX 1650.
# --skip-hf-offline reuses the (slow, unchanged) HF offline rows already in results/.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p results
$PY scripts/bench_gemm_dispatch.py results/gemm_dispatch.json
$PY scripts/bench_attention.py results/attention.json
$PY bench/correctness.py results/correctness_fp16.json
if [ "$1" != "--skip-hf-offline" ]; then rm -f results/offline.jsonl; $PY bench/offline.py --all --n 64
else python3 -c "
import json; rows=[json.loads(l) for l in open('results/offline.jsonl')]
open('results/offline.jsonl','w').write(''.join(json.dumps(r)+'\n' for r in rows if r['arm'].startswith(('hf', 'vllm'))))"
  for a in nanoserve-nokernel nanoserve; do $PY bench/offline.py --arm $a --n 64; done; fi
# vLLM baseline needs its own venv (it pins its own torch): VLLM_PY=/path/to/venv/bin/python
if [ -n "$VLLM_PY" ]; then $VLLM_PY bench/vllm_offline.py --out results/offline.jsonl; fi
$PY scripts/step_breakdown.py
$PY scripts/bench_steps.py results/steps.json
$PY bench/spec.py results/spec.json
rm -f results/online.jsonl
for a in nanoserve nanoserve-seq hf-server; do $PY bench/online.py --arm $a --rates 1 2 4 --n 40; done
python3 scripts/make_tables.py --write
echo ALL-BENCHMARKS-DONE
