# Notebook index

Last synchronized: November 17, 2026.

The notebooks below include the latest saved MIPDS work and course reference
notebooks for previous weeks. Course notebooks can contain the author's example
outputs; those outputs are not evidence of a new run by Aaron.

| Week / activity | Notebook | Source and status |
| --- | --- | --- |
| Setup | [mipds_setup_check.ipynb](mipds_setup_check.ipynb) | Existing repository copy retained; all seven code cells have saved execution counts and no error outputs. This is newer than the unexecuted September 3 Library copy. |
| Chapter 1: feature engineering and learning | [hands_on_ch1.ipynb](hands_on_ch1.ipynb) | Current course reference snapshot; any saved example results are inherited from the source. |
| Chapter 2: activation functions | No standalone notebook | The XOR/spiral activity uses TensorFlow Playground. No Chapter 2 notebook was found in the course repository or saved work. |
| Chapter 3: training dynamics | [hands_on_ch3.ipynb](hands_on_ch3.ipynb) | Current course reference snapshot, matching the earlier supplied activity. Its offline experiments use synthetic data and an MLP; the optional CIFAR-10 CNN cell is disabled by default. |
| Chapter 4: architecture exploration | [hands_on_ch4.ipynb](hands_on_ch4.ipynb) | Original course notebook. Its inherited saved run reports failed weight downloads and an untrained fallback. Use the executed local version below for the saved successful run. |
| Chapter 4: latest successful saved run | [hands_on_ch4_executed_local.ipynb](hands_on_ch4_executed_local.ipynb) | Latest separately saved local execution, September 21. Outputs confirm genuine ImageNet ResNet-50 and EfficientNet-B0 weights. The code matches the original; this synthetic exploration is separate from Milestone 4 transfer training. |
| Milestone 4: vision backbone installation | [Recovered run](Aaron_Tobias_MIPDS_Milestone_4_Executed_Recovered.ipynb) · [original scaffold](Aaron_Tobias_MIPDS_Milestone_4_Video_Training.ipynb) | Keras ResNet-50 training logs, validation metrics, and 20-frame audit table. Journal observations remain in progress. |
| Chapter 5: detection exploration | [hands_on_ch5.ipynb](hands_on_ch5.ipynb) | Current course reference snapshot. Code matches the supplied Chapter 5 notebook; unexecuted. |
| Milestone 5: detection and spatial features | [Reviewed run](Aaron_Tobias_MIPDS_Milestone_5_Reviewed.ipynb) · [original scaffold](Aaron_Tobias_MIPDS_Milestone_5_Working_Notebook.ipynb) | Saved threshold and pooling results with written review. A final clean rerun after that review is pending. |
| Milestone 6: vision audit | [Completed audit notebook](Aaron_Tobias_MIPDS_Milestone_6_Vision_Audit.ipynb) | Saved 570-frame evaluation, 60.00% top-1 accuracy, completed error review, Model Card, and Interface Specification. Figure outputs are in the original notebook export. |

## Continue from the recovered vision run

Use the existing Milestone 4 checkpoint and the original split to follow the
[Milestone 6 workflow](../docs/VISION_RECOVERY_WORKFLOW.md). The public notebook
copies retain the numerical evidence and written work. The original notebook
exports contain the frame previews, heatmaps, and detector overlays.

## Course attribution and source versions

Course material: **Moody Amakobe, _Deep Learning: A Comprehensive Guide_**
(2026), [proff-amakobe/oer-books](https://github.com/proff-amakobe/oer-books/tree/6ea2345f4c1aa7fc25cfb64eec546859c159af0e/deep-learning/material).
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
the supplied license notice is in [COURSE_LICENSE.txt](COURSE_LICENSE.txt).

- Chapters 1, 3, and 5 are copied from the course repository at commit
  `6ea2345f4c1aa7fc25cfb64eec546859c159af0e`, retaining their source code and bundled outputs.
- The Chapter 4 original is the previously saved course notebook; its code
  matches the current course source at that same commit.
- `hands_on_ch4_executed_local.ipynb` retains Aaron's saved local execution
  outputs, with the course code unchanged.
- The customized setup and Milestone 4/5 notebooks retain their latest saved
  code. The milestone notebooks adapt the exercise activities to MIPDS and
  include their own implementation references and dataset attribution.

No completed milestone notebooks for Weeks 1–3 were found: those milestone
deliverables were design documents. The setup and hands-on notebooks above
cover the notebook work located for those weeks.
