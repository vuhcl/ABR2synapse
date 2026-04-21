# Practicum talk — speaker script (~20 min)

Use this document as plain text while you speak. The audience sees only the slides; everything you say should make sense from what is on the slide plus your spoken gloss. Approximate times add to about twenty minutes; trim feature slides first if you run long.

---

## What listeners should remember

1. You joined two hearing-lab datasets to a synapse-count target from histology.
2. You turned ABR waveforms into table features (height, steepness, wobble, timing, corner sharpness).
3. You used two steps: first guess noise exposure class, then predict synapse counts.
4. You compared linear, tree, boosted-tree, tabular neural, and waveform-plus-tabular models.
5. You judged them with RMSE (error in synapse units) and R squared (versus always guessing the average).

---

## What to stress vs skim

Biology: synapse count here means “how many connections show up in the published imaging,” not a full ear lecture.

Liberman slide: two strains, controlled noise and recovery, hearing test plus tissue on the same animals.

Buran slide: paper had fifty-seven mice in four age or noise groups; many frequencies and loudness levels; you align methods so comparisons are fair.

Features: each slide is one intuitive shape idea. Skip equations unless someone asks.

Wide versus long: long is one row per loudness; wide spreads loudness across columns so one row is animal plus tone. Missing loudness cells get filled along the loudness direction. The even-loudness variant on the slide is about keeping a grid where every animal has the same columns when odd steps were not always recorded.

Stage one: classifier for loud exposure versus control from table features; its output feeds stage two.

Stage two: one plain analogy per model plus one trade-off.

Results slide: read the picture (lower is better, clusters by model family). Do not read every bar.

---

## Slide 1 — Title (about 0:45)

Thanks for being here. This work is machine learning on spreadsheets and waveforms: we ask whether a lab synapse count can be predicted from hearing-test curves that were already collected in two different animal studies.

Tonight I focus on what the pipeline does and how models were compared. I am not the right person for a full ear-biology lecture; I can point to the papers afterward.

---

## Slide 2 — Dataset Liberman, Wu 2024 (about 1:00)

First dataset is a published noise study with two mouse strains, different noise levels and recovery times, then hearing measurements and a tissue sample on the same animals.

For us it is mostly integration: we line up animal, test tone, loudness, recorded waveform, and microscope count in one table. After cleaning, the wide-table side is on the order of one hundred twenty-five animals and about five hundred animal-frequency rows. If the slide shows different rounded numbers, follow the slide.

---

## Slide 3 — Dataset Buran, JARO 2025 (about 0:55)

Second dataset is Brad Buran’s paper: hearing tests paired with counts from imaging. The paper reports fifty-seven mice in four groups: about seventeen young, thirteen acute noise, fourteen aged, thirteen aged after noise, with an ABR grid of seven test frequencies and fifteen loudness steps from ten to eighty dB.

We keep both labs in similar pipelines so model comparison stays fair. Our cleaned wide tables for this pipeline are on the order of seventy-seven animals and two hundred eighty-three rows. If the slide shows exact counts, use those.

---

## Slide 4 — Feature Amplitude (about 1:00)

Amplitude is the size of the first major upward bump in the brainstem response, relative to the dip right after. In plain language: how big does the early response look at this beep level. It is the simplest thing people have looked at historically.

The slide figure illustrates it; do not compete with the figure.

---

## Slide 5 — Feature Slope (about 1:00)

Slope is not how we turned up the volume on the machine. It is how steeply the voltage moves between that bump and the dip: a sharp corner versus a lazy ramp. Two traces can have similar height but different sharpness.

---

## Slide 6 — Feature Total variance (about 0:55)

Total variance is how much the trace wiggles around its average inside the window: energy of jitter. It catches texture that height alone misses.

---

## Slide 7 — Feature Distance (about 1:00)

Distance is how many milliseconds separate the bump and the dip. It pairs with amplitude when we think about steepness over time. Again, shape, not a new biological claim by itself.

---

## Slide 8 — Curvature Wave I peak P1 (about 0:50)

Curvature at the peak asks whether the top of the bump is needle-sharp or rounded. We summarize that corner tightness as numbers the model can read as columns.

---

## Slide 9 — Curvature Wave I trough N1 (about 0:55)

Same idea at the following valley. The downward corner can differ from the upward one even when overall size looks similar, so the model gets both.

---

## Slide 10 — Wide versus long tables (about 1:15)

Machine learning likes tidy rectangles.

Long format: each row is one animal, one test tone, one loudness. Wide format spreads loudness across columns so one row is animal plus tone with a wide feature vector. Some models prefer that layout.

When a few loudness cells are missing, we fill along the loudness direction with a simple smooth fill.

The slide also mentions an even-loudness grid for benchmarks: only even decibel steps so every animal shares the same columns, because odd steps were not always recorded for everyone. Treat that as a labeled variant on the benchmark chart, not the only preprocessing path in the project.

---

## Slide 11 — Stage 1 noise versus control (about 0:45)

Stage one is a warm-up classifier: from table features only, guess high noise exposure versus low or control. It checks that waveform summaries still carry condition information. Its output feeds stage two.

---

## Slide 12 — Preprocessing (about 0:30)

Match the two bullets on the slide. First, we z-score numeric columns so they sit on comparable scales. Second, for skewed columns we apply log one plus x, then z-score again, so heavy-tailed inputs do not dominate the model.

---

## Slide 13 — Stage 1 performance (about 0:50)

We compared logistic regression with cross-validation to a tuned random forest; we used the forest for probabilities, then averaged those to one noise score per animal for the next stage.

If the slide shows numbers for Liberman, say them as printed: cross-validation accuracy about 0.60, animal-level test accuracy about 0.86, AUC about 0.89. If the slide instead shows Brad-only numbers, use those: about 0.63, 0.75, and 1.00 for the same three metrics. If the slide shows a figure, point to the legend and the best cluster rather than reading every pixel.

---

## Slide 14 — Linear regression OLS (about 0:55)

Ordinary least squares is the best straight-line blend of features baseline. Buran’s paper leaned on linear regression for main comparisons, so we keep linear models as the reference for what a simple paper-style model does.

We run a sparse baseline, essentially one loudness of the response, and a full model with everything including the stage-one noise signal. Full is not always better when many columns are correlated.

If you put numbers on the slide for Brad test data: baseline about R squared 0.37 and RMSE about 3.8 synapses; full about R squared 0.02 and RMSE about 4.7. For Liberman test data on the same comparison: baseline about R squared 0.23 and RMSE about 3.1; full about R squared 0.43 and RMSE about 2.7. Read exactly what you printed on the slide if it differs slightly.

---

## Slide 15 — Random Forest (about 0:50)

Random forest: many decision trees vote. Good at curved relationships and interactions without hand-writing feature times feature. Trade-off: harder to tell a story than a line, and depth must be controlled for sample size.

---

## Slide 16 — XGBoost (about 0:50)

XGBoost is boosted trees: each new tree tries to fix the mistakes of the stack so far. Often strong on tables. Trade-off: more knobs to tune, still not a paragraph you read to a patient.

---

## Slide 17 — MLP (about 0:50)

MLP is a small vanilla neural network on the same spreadsheet features: layers of weighted sums with nonlinearities. More flexible, less interpretable; needs regularization so it does not memorize mice.

---

## Slide 18 — CNN plus tabular fusion (about 0:50)

A small convolutional net looks at the actual voltage trace while a parallel branch still reads the table features, then they merge for a prediction. It can catch timing patterns summaries miss; it is the most data- and preprocessing-sensitive part of the stack.

---

## Slide 19 — RMSE and R squared (about 1:05)

We report two scores that answer different questions.

RMSE is root mean squared error: typical prediction mistake in the same units as the synapse count, and large mistakes count extra. We foreground RMSE partly to stay comparable in spirit to regression-style reporting in the Buran paper: people can relate to synapses off as a ruler.

R squared is what fraction of overall spread we explain versus always guessing the average. It is scale-free and good for ranking models when difficulty differs between test sets. It can go negative if you are worse than the mean; that is not a bug.

---

## Slide 20 — Benchmark RMSE (about 1:30)

This is the summary picture: lower is better. I will not read every bar. Look for clusters: linear versus tree versus neural, and which lab’s test strip looks harder.

If the legend shows training scenarios, read it once in plain words, for example training on one lab and testing on the other as a stress test for generalization.

Optional anchors if they match your exported figure: on Liberman test, one strong row was XGB with R squared about 0.66 and RMSE about 2.1 synapses on the scenario marked on the chart. On Brad test, one strong row was XGB with R squared about 0.57 and RMSE about 3.1. If the bars do not match these anchors, trust the slide you exported.

---

## Slide 21 — Thank you and questions (about 1:15)

Three honest limits: animal lab data, not humans; rows nest inside animals so naive sample size is inflated; labels and preprocessing differ by lab.

My lane is reproducible code and benchmarks; for biology depth, the original papers and lab collaborators are the right next step.

Thanks; questions welcome, especially on implementation.

---

## Timing check

Rough targets: title 0:45; data slides about two minutes total; six feature slides about five to six minutes total; pivot about 1:15; stage one block about 2:15; five model slides about four minutes; metrics about 1:05; results about 1:30; thanks about 1:15. Total near twenty minutes.

If you run long, cut twenty seconds from each feature slide first, then shorten each model analogy by about ten seconds.

---

## Q and A stash not for the slide

244 points and peak finder: we standardize length so the same code runs on every row, then find wave one automatically and snap peaks to the nearest sample.

Noise prediction versus true label: training can use the true group when stage one never saw that animal; say only if asked.

Why not accuracy for synapse regression: counts are continuous; RMSE and R squared are standard. Accuracy is for stage one categories.
