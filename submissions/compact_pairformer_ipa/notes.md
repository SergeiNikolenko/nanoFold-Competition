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
- hardware: pending official run
- wall_clock_time: pending official run
- commit: pending submission commit

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
