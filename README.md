# Assignment 2 — Linear Models and SVMs

Team implementation for Introduction to Machine Learning.

We implement linear regression, ridge regression, logistic regression, softmax, and SVMs using NumPy, then run the comparisons requested in the assignment PDF.

## Current progress

Parts A and B include implementations of the required numerical experiments and checks. Verification uses temporary output folders during development. Written derivations and explanations will be prepared after the code and experiments are finalized. Parts C, D, E and F still need the corrections and completion described below.

Task numbers below refer to **Section 6 — Tasks** in the assignment PDF. The PDF jumps from Task 28 to Task 30.

## Work allocation

| Member | Scope | Files |
| --- | --- | --- |
| Mostafa | Parts A and B, Tasks 1–10 | `part_a.py`, `part_b.py` |
| Member 2 | Parts C and D, Tasks 11–22 | `part_c.py`, `part_d.py` |
| Member 3 | Parts E and F, Tasks 23–28 and 30–31, plus notebook assembly and submission | `part_e.py` and the final notebook |

Each member provides their own derivations, check results, figures, tables, and interpretations. Member 3 assembles Part F using contributions from everyone, organizes the final notebook, and prepares the PDF and ZIP.

The assignment specifies a four-person team. Confirm the current three-person arrangement with the teaching staff.

## Files

```text
src/
    assignment2_core.py    Shared mathematical functions and optimizers
    assignment2_runner.py Command-line entry point
    part_a.py             Data loading, preprocessing, and toolkit checks
    part_b.py             Linear and ridge regression experiments
    part_c.py             Logistic regression experiments and comparisons
    part_d.py             Softmax experiments and comparisons
    part_e.py             SVM experiments and comparisons
    plot_utils.py         Figure-saving helper
requirements.txt          Python dependencies
README.md                 Team guide
.gitignore                Local files excluded from Git
```

Part F will be written in the final notebook once the experiments are complete.

Current work focuses on the source code. Generate and review results when testing the experiments, then assemble the final notebook with the team.

## Run the code

Run these commands from the project folder using Python 3.9 or newer with compatible package versions.

```bash
python -m pip install -r requirements.txt
python src/assignment2_runner.py --part a
```

Choose `a`, `b`, `c`, `d`, or `e` to run a single Part. To run all implemented experiments:

```bash
python src/assignment2_runner.py
```

The runner creates `outputs/results/` when it saves figures. For environments without interactive plotting windows, set `MPLBACKEND=Agg` before running.

The last verified run used Python 3.13. The four datasets come with scikit-learn and are loaded by the code.

## Where to start

- **Mostafa:** review the Parts A and B source code, then test the experiments and prepare the results for final notebook assembly.
- **Member 2:** correct logistic validation-loss selection and use the λ = 0.01 model in Task 15. Complete the logistic boundary, softmax initialization and final-model error analysis, the two-class equivalence check, and the two-feature Iris experiment.
- **Member 3:** finish SVM validation curves and objective/margin/support-vector comparisons. Add multiclass hinge training and the required kernel figures. Correct Task 28 to use training rows only for cross-validation. Assemble Part F and the final deliverables after receiving the other members' work.

The PDF contains the full requirements and tolerances for each task. Bonus tasks are outside the current allocation.

## Working together

Use a separate branch for each member. Work in your assigned Part files and coordinate changes to shared functions in `assignment2_core.py`.

Select model settings using validation data or the specified cross-validation procedure. Use scikit-learn in the places allowed by the PDF.

When handing over a Part, include the code, required checks, figures or tables, and written interpretation. Record any unfinished tasks or failed checks so they can be resolved before final assembly.
