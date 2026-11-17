# Vision recovery and Milestone 6 workflow

The [Architecture Record](ARCHITECTURE_RECORD.md) reconciles the earlier design with the recovered vision run. The repository includes Milestone 4 and 5 copies with the saved logs, numerical results, and written analysis. The full notebook exports contain the original frame-based figures for course review.

| Milestone | Current evidence | Remaining work |
| --- | --- | --- |
| 4 · Backbone | Recovered training output, validation audit, run record, and split manifest | Finish the journal observations from the original audit images |
| 5 · Detection | Reviewed threshold sweep, heatmap observations, and two pooling probes | Verify a clean rerun after the written review if preparing a submission |
| 6 · Vision audit | Prepared notebook, metrics export, and Model Card and Interface Specification templates | Run the original checkpoint on the test split and complete the visual review |

The original Milestone 5 assignment uses 20 domain images, confidence thresholds of 0.25, 0.50, and 0.75, and concatenated pooling at 1 × 1, 2 × 2, and 4 × 4. Use these settings to reproduce the reviewed experiment. The separate YOLO detector uses pretrained COCO labels.

## Run Milestone 6 in Colab

1. Open the [Vision Audit notebook](../notebooks/Aaron_Tobias_MIPDS_Milestone_6_Vision_Audit.ipynb). In Colab, set `PROJECT_DIR` to the full path of the existing project folder. A path under `/content/drive/MyDrive/` mounts Drive in your Colab session.
2. Use the original `classification_manifest.csv`, `milestone4_outputs/m4_run_record.json`, `milestone4_outputs/m4_finetuned.keras`, and the 570 test PNGs at their relative paths under `frames/`. The notebook checks split membership across the full manifest and verifies the checkpoint and each test image against the recorded hashes. Colab can read the existing checkpoint where it is stored.
3. Run the cells in order. The evaluation uses the saved checkpoint, original class order, and unchanged test split. It exports frame-level metrics, the confusion matrix, confidence calibration, up to 30 sampled mistakes, GradCAM overlays, t-SNE, and nearest neighbor comparisons to a versioned folder under `milestone6_outputs/`.
4. Inspect the figures and enter the error and representation observations in the marked cells. Confirm the training date from the project evidence. Rerun the report and export cells. Use the measured results to complete the Model Card and Interface Specification.
5. Download the executed notebook with its rendered outputs and the generated reports. Confirm the reports cover the original assignment requirements before marking the interface frozen for language and fusion work.

For reruns of the training notebook, choose a fresh project directory. Evaluation uses the previously saved checkpoint and performs inference with the model frozen.
