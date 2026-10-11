# Notebook index

Last synchronized: October 11, 2026.

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
| Milestone 4: vision backbone installation | [Aaron_Tobias_MIPDS_Milestone_4_Video_Training.ipynb](Aaron_Tobias_MIPDS_Milestone_4_Video_Training.ipynb) | Latest October 11 public exercise-recognition workflow using Keras ResNet-50, seven exercise labels, participant-disjoint splits, frozen-head training, selective fine-tuning, and a 20-image audit. Prepared and unexecuted. |
| Chapter 5: detection exploration | [hands_on_ch5.ipynb](hands_on_ch5.ipynb) | Current course reference snapshot. Code matches the supplied Chapter 5 notebook; unexecuted. |
| Milestone 5: detection and spatial features | [Aaron_Tobias_MIPDS_Milestone_5_Working_Notebook.ipynb](Aaron_Tobias_MIPDS_Milestone_5_Working_Notebook.ipynb) | Latest October 11 notebook, connected to the actual Milestone 4 export and participant split. Prepared and unexecuted. |

## Start the current milestone workflow

1. Open [Milestone 4 in Colab](https://colab.research.google.com/github/Throwbias/mipds-deep-learning/blob/main/notebooks/Aaron_Tobias_MIPDS_Milestone_4_Video_Training.ipynb), select a GPU runtime, and run the cells in order.
2. Preserve its dataset frames, checkpoint, preprocessing record, and manifests.
   Both notebooks default to `/content/mipds_public_exercises`; use the same
   persistent project folder when moving between Colab sessions.
3. Open [Milestone 5 in Colab](https://colab.research.google.com/github/Throwbias/mipds-deep-learning/blob/main/notebooks/Aaron_Tobias_MIPDS_Milestone_5_Working_Notebook.ipynb) and supply the additional scenes requested by its input manifest.
4. Fill the audit and journal responses from actual rendered results, and save
   the executed notebooks back to this repository.

The Milestone 4 and 5 files are code scaffolds awaiting real training and
evaluation. Uploading them does not complete either milestone.

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
