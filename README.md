# Drug repurposing with Heterogeneous Graph Transformers

Link prediction of `drug —indication→ disease` edges on a large biomedical knowledge graph with a Heterogeneous Graph Transformer (PyTorch Geometric `HGTConv`), followed by an LLM-assisted ADMET/safety prioritisation of the predicted candidate drugs. Applied to finding repurposing candidates for Nipah virus.

This repository is a showcase of the code. It contains **no data, no trained models and no results**: the knowledge graph is downloaded separately, candidate-drug lists belong to ongoing collaborative work, and the ADMET stage depends on licensed DrugBank data that cannot be redistributed. Notebooks are shipped with their outputs cleared. Treat the repo as something to read, not something to run end to end.

## Origin and what is new

This project started from the public HGTDR code and paper, *"HGTDR: Advancing Drug Repurposing with Heterogeneous Graph Transformers"* (data: <https://dml.ir/HGTDR_data>). The upstream release is a single notebook plus a small `utils.py` that holds the HGT model, data preparation, batching and train loop for drug–disease indication prediction. I re-engineered it into a configurable, scripted pipeline and extended it to a new disease (Nipah virus) and to candidate prioritisation. The core HGT forward pass and the binary loss are retained from upstream; most of what surrounds them (data preparation, sampling, batching, splits, CV, HPO, prediction) was rewritten or is entirely new. The upstream authors' work is credited here, and their data release is required to run anything.

Re-engineered or new relative to upstream:

- **Scripted, config-driven pipeline** instead of a notebook, with a config-driven `HGT` (layer count, heads, optional residual connections with per-type projections, configurable node and relation types) and a multi-class loss/predictor. Argparse CLIs for graph building, training, cross-validation, HPO and prediction (`src/`).
- **Supervision/message-passing edge split** with per-batch supervised edges and generated negative edges (`utils_sampling_processing.py`), and a neighbour-sampling setup (`HGTLoader`) whose parameters were chosen by a coverage study (see the notebooks).
- **Cross-validation with inner folds for tuning** (outer CV folds, each split again into inner folds that the Bayesian hyperparameter optimisation (`scikit-optimize`) averages over), run in parallel over outer folds (`HPO.py`, `HPO_parallel.py`, `run_cv.py`).
- **Merging of contraindications into the label set**, and a 3-class variant (indication / contraindication / none).
- **Prediction for a new disease (Nipah virus)**: graphs augmented with condition-specific gene sets, scored with the trained CV models, cross-referenced against approved drugs, and compared across folds (`nipah_pred.py`, `pred_cv.py`).
- **ADMET/safety prioritisation with LLMs** over the predicted candidates (`src/admet/`, `src/admet_scoring.ipynb`).

## Method in brief

- **Task.** Heterogeneous graph with ~10 node types and ~48 edge types (about 8.1M edges counting reverse edges; the default relation filter keeps about 4 node types and 22 edge types). The supervised relation is `drug —indication→ disease`. Node and relation types are configurable (`--node_type1/--node_type2/--rel`).
- **Model** (`model.py`). Per-node-type input projections, stacked `HGTConv` layers with optional residuals, and concatenation of every layer's output before a final linear layer. An MLP predictor scores concatenated drug and disease embeddings; the loss is class-weighted BCE (or weighted cross-entropy for the 3-class variant).
- **Sampling.** The full graph is too large to train on directly and training on the full network performed much worse than on sampled subgraphs in the experiments run, so training uses `HGTLoader` neighbour sampling. The number of layers, batch size and samples per layer were chosen by measuring how much of each relation the sampled batches cover over many epochs.
- **Evaluation.** AUROC and AUPR from cross-validated runs are in the range of roughly 0.94–0.97; the final configuration reached about 0.975 AUPR/AUROC on a held-out outer fold. Metrics vary by about ±0.005 between evaluations of the same model because negatives and sampled subgraphs are random.

## Repository layout

```
src/
  preprocess_KG.py              triplets -> HeteroData graphs; relation filtering; label merging; gene-set graphs
  model.py                      HGT, MLPPredictor, loss
  utils_sampling_processing.py  data split, negative edges, HGTLoader sampling, batch processing
  utils_metrics.py              AUROC/AUPR, plots, logging
  run_training.py               single training run (also the `run` function used by HPO)
  run_cv.py                     cross-validated training
  HPO.py, HPO_parallel.py       Bayesian HPO (skopt) over one CV set / one subprocess per fold
  SpaceHPO_*.json               HPO search spaces
  nipah_pred.py, pred_cv.py     candidate-drug prediction with the CV models and cross-fold overlap analysis
  admet/                        ADMET/safety annotation and scoring pipeline (see below)
  *.ipynb                       analysis notebooks (see below)
```

Scripts use relative paths (`../data`, `../out`) and are meant to be run from `src/`.

### Typical commands

```bash
# 1. Build HeteroData graph(s) from the KG triplets + entity embeddings
python preprocess_KG.py --data_directory ../data --save_path ../data/merged

# 2. Single training run
python run_training.py --data_path ../data --model_path ../out

# 3. Cross-validated training, then prediction for the target disease
python run_cv.py --data_path ../data/merged --out_path ../out/merged --batch_size 134 --workers 1
python pred_cv.py --CVs_dir ../out/merged --data_dir ../data/merged --batch_size 134 --workers 1

# 4. Bayesian HPO
python HPO.py --data_path <CVdir> --out_path <out> --metric AUPR --n_calls 50
python HPO_parallel.py --data_path <dir containing CV*/> --out_path <out>
```

### Data

The knowledge graph (`kg.csv`), the entity dictionary and the node embeddings come from the upstream project's data release (<https://dml.ir/HGTDR_data>). The gene sets used to build the Nipah graphs come from collaborators and are not included.

### Environment

See `requirements.txt` (torch 1.13.1, torch-geometric 2.3.1 and the other upstream pins, plus `scikit-optimize` for HPO and `instructor`/`openai`/`pydantic` for the ADMET part).

## Notebooks

Two working notebooks are included. Their outputs are cleared, so they show the analysis code, not the results, and they expect the data layout above. Earlier prototype notebooks were left out because their logic now lives in the scripts.

- **`sampling_overlap&exclusivity.ipynb`**: the experiment behind the sampling parameters. For different (layers, batch size, samples-per-layer) settings it measures, over hundreds of epochs, how many edges of each relation type are covered by the sampled subgraphs and how much the subgraphs of different batches overlap. The motivation: a configuration that looks cheap can leave large parts of dense relations (protein–protein, phenotype–phenotype) essentially unseen. The analysis helpers (coverage, exclusivity, per-batch means, plots) are only in this notebook, not in the scripts.
- **`OpSampling.ipynb`**: working notebook for trying sampling parameters (samples per layer, batch size) and comparing the relation composition and average degrees of sampled subgraphs. Its `sampling` and `process_batch` are near copies of the script versions; the edge-count, degree and diff helpers are notebook-only.

The ADMET pipeline has its own notebook, described next.

## ADMET/safety prioritisation (LLM pipeline)

After prediction, candidate drugs are ranked for practical suitability using DrugBank data and a **locally hosted** LLM (Qwen 2.5 served by Ollama), so that licensed DrugBank text never leaves the machine.

- **`admet_scoring.ipynb`** (first version of the pipeline, superseded by `src/admet/` below): a two-agent design built with `instructor` and pydantic schemas. Agent 1 turns pre-grouped DrugBank fields into a structured ADMET summary; agent 2 turns that summary into a structured evaluation used to rank drugs. The run checkpoints to JSON so it can resume, and the notebook then aggregates repeated runs.
- **`src/admet/`**: a redesign of that pipeline, structured as three steps.
  1. `build_inputs.py` parses the DrugBank XML into one table per candidate and computes deterministic flags (type, ATC codes, categories, routes) in Python.
  2. `annotate.py` makes one LLM call per drug with a pydantic schema, to write human-readable fields and evidence-backed annotations. `verify.py` then removes anything not supported by the input (quotes that can't be found, numbers that don't appear, tiers with no supporting text).
  3. `report.py` computes the final score as the model score plus points from an editable table (`scoring_points.csv`), so the ranking itself is not decided by the LLM.
  - Optional fetchers (`add_fda_labels.py`, `add_chembl.py`) add openFDA label text and ChEMBL potency. `smoke_test.py` and `test_fetchers.py` run on synthetic data with a stub in place of the LLM.

**This part cannot be run from this repository.** It needs the DrugBank XML (a licensed download, not redistributable), a running Ollama server with the model pulled, and the candidate-drug table from the prediction step, none of which are included. It is here as a demonstration of the design. To keep the repo clean of licensed or unpublished material, two files are also left out: the few-shot example prompt (`annotate_fewshot.txt`, so use `--no_fewshot`) and the manual drug-name alias list (`aliases.csv`, optional). Paths of the form `<path/to/drugbank_full_database/...>` are placeholders.

## License

MIT, see `LICENSE`. Code derived from the upstream HGTDR project remains the work of its authors; see "Origin and what is new".
