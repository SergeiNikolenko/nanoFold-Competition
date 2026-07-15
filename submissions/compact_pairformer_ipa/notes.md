## Submission Summary

`compact_pairformer_ipa` is a train-from-scratch, atom14 protein-folding model for the official
`limited` track. It compresses the official MSA with a short Evoformer stack, discards the deep MSA
state after the query row, and continues with a compact pair/single trunk before the proven
minAlphaFold2 IPA structure module.

The model has:

- two MSA Evoformer blocks over the official MSA features
- eight Pairformer-style blocks with triangle geometry updates
- QK-normalized, pair-biased single attention
- ReLU-squared single transitions and zero-initialized residual outputs
- a four-layer IPA structure module with stereochemical atom14 construction
- Muon updates for hidden trunk matrices and Adam updates for embeddings, gates, heads, norms, and
  structure parameters
- BF16 trunk execution on CUDA with the AlphaFold loss evaluated from FP32 tensors

The Muon implementation follows the public Muon method by Keller Jordan and collaborators. The model
uses no pretrained parameters from Muon, OpenFold3, Protenix, AlphaFold, or any other external model.

## Method Rationale

The architecture keeps the official MSA signal but spends most model depth on pair and single
representations. This follows the data-efficient direction of AlphaFold3-style systems without
bringing their external distillation data, templates, pretrained weights, or full diffusion sampler
into the benchmark.

The hidden rank metric is area under the fixed-sample learning curve. The loss therefore blends from
the initial AlphaFold objective into the geometry-aware fine-tuning objective between steps 1,000 and
5,000, while the learning-rate schedule remains independent of that handoff. This makes backbone,
side-chain, and stereochemical supervision active for the high-weight middle checkpoint rather than
only near the end of the run.

## Competition Compliance Checklist

- [x] Used only the provided benchmark data; no external structures or labels are loaded.
- [x] Uses no external structures, pretrained weights, model embeddings, template lookup, or network
  access.
- [x] Kept dataset manifests fixed at `data/manifests/train.txt` and `data/manifests/val.txt`.
- [x] Ignores template tensors and does not run template search.
- [x] Model outputs atom14 coordinates as `pred_atom14` with shape `(B, L, 14, 3)` in Angstroms.
- [x] Inference does not read supervision labels.

## Required Run Metadata (`limited`)

- max_steps: 30000
- effective_batch_size: 8
- sample_budget: 240000
- residue_budget: 61440000
- crop_size: 256
- MSA depth: 192
- seed: 0
- trainable_parameters: 6024402
- hardware: NVIDIA A100-PCIE-40GB 40 GB (CUDA 12.8, PyTorch 2.7.1; full-shape smoke)
- wall_clock_time: 11 seconds for four full-shape smoke train steps; full 30000-step run pending
- commit: 656775bc0d4519962aac290e0f518050b90db665

## Validation Evidence

The committed source was exercised on an A100 with the official crop size and MSA depth
(`B=1, L=256, N=192`). A BF16 forward, backward, gradient clip, and Muon/Adam optimizer step used
2.60 GiB peak allocated CUDA memory and 2.96 GiB peak reserved memory. All gradient tensors were
finite, and the step completed in 1.82 seconds after model construction.

An official-mode four-step synthetic smoke run wrote checkpoints at steps 0, 2, and 4. Resuming
from step 2 reproduced the step-3 and step-4 losses exactly. Label-free multi-checkpoint prediction
and the separate FoldScore process both completed. The synthetic score is intentionally not reported
as a benchmark result; the full public-data learning curve and sealed hidden score remain pending.

## How to run

```bash
python scripts/validate_submission.py \
  --submission submissions/compact_pairformer_ipa \
  --track limited \
  --strict

python train.py \
  --config submissions/compact_pairformer_ipa/config.yaml \
  --track limited \
  --official
```
