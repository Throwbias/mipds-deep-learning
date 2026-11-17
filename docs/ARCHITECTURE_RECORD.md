# MIPDS Architecture Record

Aaron Tobias · CSC 5352 Deep Learning · Recovered vision baseline · November 17, 2026

## Purpose

MIPDS supports trainer review of exercise recordings. The original prototype proposed estimating projected knee angles from side-view squat recordings using MediaPipe landmarks, timestamps, and tracking quality flags. MediaPipe and temporal analysis remain planned. The implemented coursework baseline recognizes seven exercise categories from side-view still frames, with potential use in organizing recordings for review.

The original design decisions are preserved in the [Design Philosophy](architecture_sources/Aaron_Tobias_MIPDS_Design_Philosophy.docx), [Perception Layer ADR](architecture_sources/Aaron_Tobias_MIPDS_Milestone_2_Perception_Layer_ADR_Formatted.docx), and [Training Engine Design](architecture_sources/Aaron_Tobias_MIPDS_Architecture_Record_Section_3_Training_Engine.docx).

## Current vision interfaces

| Component | Interface and semantics |
| --- | --- |
| Classifier input | RGB image. Resize the whole image to 224 × 224 with bilinear interpolation. Apply Keras ResNet-50 preprocessing outside the checkpoint, converting RGB to BGR and subtracting channel means of 103.939, 116.779, and 123.68 from unscaled pixel values. |
| Classifier output | Seven softmax scores in the run record's class order. The top score is a model score for exercise type. Confidence calibration is part of the held-out audit. |
| Pooled feature vector | 2,048 float values from global average pooling. This tensor feeds the trained classification head. |
| Spatial feature map | 7 × 7 × 2,048 channel-last activations from ResNet-50 layer `conv5_block3_out` at 224 × 224 input. |
| Milestone 5 comparison | A separate linear probe uses a 43,008-value vector formed from flattened 1 × 1, 2 × 2, and 4 × 4 adaptive average pooling. The frozen classifier retains its original global average pooling head. |
| Milestone 5 detection | The separate COCO-pretrained YOLOv8-nano model returns box corners in oriented original-image pixels, COCO labels, and confidence scores. The provisional threshold for human review is 0.50. |

The classifier's order is `dumbbell_side_bend`, `bent_over_row`, `bicep_curl`, `push_up`, `shoulder_press`, `squat`, `triceps_extension`.

## Data and training

Version 3 of the public exercise dataset supplies the clips and exercise labels. The run samples at most 12 frames from the central 80% of each side-view clip. Frames inherit their clip's exercise label. Source form labels serve as metadata. Dataset attribution and the CC BY 4.0 license are recorded in the training notebook. [Mendeley Data, DOI 10.17632/kgbb3yn47p.3](https://data.mendeley.com/datasets/kgbb3yn47p/3)

| Split | Frames | Source clips | Participants |
| --- | ---: | ---: | ---: |
| Training | 2,531 | 213 | 17 |
| Validation | 559 | 47 | 4 |
| Test | 570 | 48 | 4 |

All seven labels appear in every split. The saved split passes overlap checks by participant, session ID, video path, and recorded frame hash. Frame counts include correlated observations from the same clips. Split membership and checkpoint selection are recorded in the [manifest](../experiments/m4_recovered/classification_manifest.csv) and [run record](../experiments/m4_recovered/m4_run_record.json).

| Earlier plan | Recovered run |
| --- | --- |
| Fine-grained squat-state classification | Seven exercise categories from public recordings |
| AdamW, five-epoch warmup, and a longer schedule | Adam at 0.0003 for 20 frozen-backbone epochs. Stage 2 started at 0.00003 with cosine decay and ended after nine epochs. |
| Broader augmentation | Small rotations and contrast changes during training |
| Checkpoint selection by validation macro-F1 | Lowest validation loss within each stage |

The classifier started from ImageNet ResNet-50 and used global average pooling, 0.30 dropout during training, and a seven-class Dense layer. Stage 2 unfroze the final two residual stages with BatchNormalization frozen. Saved checks found that the intended frozen weights stayed fixed and that the unfrozen weights changed. The checkpoint and preprocessing details are versioned by the SHA-256 digests in the run record.

## Recorded evaluation

| Validation measurement | Saved result |
| --- | ---: |
| Original classifier with frozen backbone | 59.93% accuracy |
| Original classifier after fine-tuning | 71.56% accuracy |
| Fine-tuned classifier macro-F1 | 0.7022 |
| Milestone 5 global-average linear probe | 70.84% accuracy |
| Milestone 5 multiscale linear probe | 74.60% accuracy |

The original classifier gained 11.63 percentage points on the same validation split. The Milestone 5 probes used their own linear heads. The detector comparison ran at 0.25, 0.50, and 0.75 on 20 domain images. The detector outputs support qualitative review. Measuring detection mAP requires labeled boxes and a separate evaluation.

## Milestone 6 handoff

The audit uses the saved classifier checkpoint on the original test split. Overall and per-class accuracy, confidence calibration, GradCAM, and representation analysis are pending execution. The target online-client population, front and diagonal views, and movement or form judgments are outside the evidence from this run.

Future fusion can consume the versioned class order, 2,048-value global vector, and 7 × 7 × 2,048 feature map once the audit is complete. Scores need a confidence policy informed by measured reliability. Pose geometry and repetition counts require separate components and evaluation. [Module 6 workflow](VISION_RECOVERY_WORKFLOW.md)
