# MIPDS — Deep Learning Coursework

This repository contains my coursework, experiments, and
Multimodal Intelligent Perception and Decision System capstone.

## Project direction

Explore exercise-video analysis for coaching, beginning with
joint landmarks and estimated two-dimensional joint angles.

## Development environment

- Ubuntu 24.04 through WSL2
- Python 3.12
- TensorFlow 2.20.0
- NVIDIA GeForce RTX 4060 Ti

## Organization

- notebooks: hands-on activities and verification
- docs: design documents and written work
- experiments: experiment notes and results
- mipds_language: note preprocessing and the language encoder
- mipds_fusion: projected feature concatenation and integration audit
- mipds_generative: conditional VAE architecture and generation contract checks

## Current notebooks

The current vision prototype recognizes seven exercise types in frames from
public side-view recordings. Its recovered Milestone 4 training and Milestone 5
representation experiments are indexed in the [notebook guide](notebooks/README.md).
The [Architecture Record](docs/ARCHITECTURE_RECORD.md) documents the baseline,
including how it evolved from the planned pose workflow.

The Milestone 6 audit evaluated the frozen classifier on 570 held-out frames.
See the [Milestone 6 workflow](docs/VISION_RECOVERY_WORKFLOW.md) for the notebook
and reproduction steps.

## Language encoder

The current language feature extractor uses pinned DistilBERT Base Uncased
weights. Its frozen adapter returns 64 × 768 token states and a 768-value
content mean for each note. The [Architecture Record](docs/ARCHITECTURE_RECORD.md)
contains the active contract and the earlier Milestone 8 prototype.

| Milestone | Submission |
| --- | --- |
| 7 | [Language preprocessing pipeline](docs/milestones/Aaron_Tobias_MIPDS_Milestone_7_Language_Preprocessing_Pipeline.docx) |
| 8 | [Language encoder integration](docs/milestones/Aaron_Tobias_MIPDS_Milestone_8_Language_Encoder_Integration.docx) |
| 9 | [Pretrained language model integration](docs/milestones/Aaron_Tobias_MIPDS_Milestone_9_Pretrained_Language_Model_Integration.docx) |
| 10 | [Multimodal fusion layer](docs/milestones/Aaron_Tobias_MIPDS_Milestone_10_Multimodal_Fusion_Layer.docx) |
| 11 | [Generative module](docs/milestones/Aaron_Tobias_MIPDS_Milestone_11_Generative_Module.docx) |

Install the Milestone 9 dependencies from `mipds_language/requirements.txt`
in an isolated Python 3.12 environment. A CPU installation of PyTorch 2.9.1
was used for the audit. Download `config.json`, `vocab.txt`, and
`model.safetensors` from revision
`12040accade4e8a0f71eabdb258fecc2e7e948be` of
`distilbert/distilbert-base-uncased`.
[Pinned model files](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be)

```bash
python -m mipds_language.pretrained_audit --model-dir /path/to/model --output /path/to/audit-results
```

The [saved Milestone 9 audit](experiments/m9_pretrained/audit.json) records 16
passing integration checks and the fictional-note readout result. The CLI
exports `audit.json` and `readout.npz`. Base weights remain retrievable from
the pinned model repository.

### Milestone 8 prototype

Milestone 8 adds a two-block Transformer that takes the Milestone 7 note tensors.
The CPU audit trains a reported-direction probe on fictional notes and exports
held-out metrics, attention samples, and contract checks. The recorded run used
NumPy 2.3.5 and Autograd 1.8.0. [Saved audit](experiments/m8_language/audit.json)

Run the audit from the repo root in a Python environment with the audited NumPy
and Autograd versions. Use the pinned BERT Base Uncased vocabulary from Milestone
7. [Vocabulary source](https://github.com/microsoft/SDNet/blob/master/bert_vocab_files/bert-base-uncased-vocab.txt)

```bash
python -m mipds_language.audit --vocab /path/to/vocab.txt --output /path/to/audit-results
```

The output directory receives `audit.json` and `language_encoder.npz`. Load the
checkpoint with NumPy and pass the tokenized notes to
`mipds_language.encoder.encode_notes`. The Architecture Record specifies the
output shapes and flags.

## Multimodal fusion

The fusion layer projects the frozen 2,048-wide vision features and 768-wide
text features to 256 channels each. It concatenates vision before text and
returns a 512-wide vector with presence and truncation flags. The callable
interface checks clip IDs, model versions, feature widths, and numeric inputs.

The [Milestone 10 audit](experiments/m10_fusion/audit.json) passed 24 checks on
seven real cached vision vectors paired with fictional notes. The recorded
fusion state uses seeded initial weights. The projections require paired-data
training before a downstream prediction score can be evaluated.

Use the Milestone 9 Python environment and pretrained model files. The vision
input is the original `test_embeddings.npz` exported by the completed
Milestone 6 notebook.

```bash
python -m mipds_fusion.audit --vision-embeddings /path/to/test_embeddings.npz --model-dir /path/to/distilbert --output /path/to/fusion-audit
```

The command exports `audit.json` and `fusion_initial_state.safetensors`.
See the [Architecture Record](docs/ARCHITECTURE_RECORD.md) for output fields
and the [design document](docs/milestones/Aaron_Tobias_MIPDS_Milestone_10_Multimodal_Fusion_Layer.docx)
for the projection and fusion decisions.

## Generative module

Milestone 11 defines a conditional VAE for 64 × 64 RGB setup thumbnails intended
for trainer review. The implemented architecture has 3,302,787 parameters.
The condition adapter combines the 512 fusion features, two presence flags,
and one truncation flag into a 128-value context. The posterior and decoder
both receive that context. Generation samples a 32-value standard-normal
latent and preserves the source clip IDs, flags, and seed.

The [architecture audit](experiments/m11_generative/audit.json) passed 30 checks
using the recorded M10 fusion state, real cached vision vectors, fictional
notes, and an analytic RGB image fixture. The temporary optimizer step is
reversed. The generator and fusion projections remain untrained on paired
examples. Image quality and conditioning fidelity have not been evaluated.

Use the same dependencies and input files as the M10 audit.

```bash
python -m mipds_generative.audit --vision-embeddings /path/to/test_embeddings.npz --model-dir /path/to/distilbert --output /path/to/generative-audit
```

The command exports `audit.json` and `generator_initial_state.safetensors`.
The [design document](docs/milestones/Aaron_Tobias_MIPDS_Milestone_11_Generative_Module.docx)
specifies the architecture, conditioning path, training plan, and limits.
