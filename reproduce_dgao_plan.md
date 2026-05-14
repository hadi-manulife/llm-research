## Plan: Reproduce DGAO Experiment

TL;DR: Reproduce the paper’s main order-fairness experiment from arXiv:2605.11974 by rebuilding the data construction pipeline, training the baseline and DGAO variants on the reported datasets, and validating against the paper’s three metrics: Accuracy, Consistency Rate, and Overconfidence Rate. The first pass should target the main experimental setup in Section 4 and Appendix B, then optionally extend to the ablations and generalization checks.

**Detailed Kaggle Setup and Run Guide**
1. Prepare Kaggle environment.
    - Create a new Kaggle Notebook.
    - In Notebook Settings, enable GPU accelerator.
    - Use internet-enabled session if you plan to download Hugging Face datasets/models at runtime.
    - Set a persistent working folder convention:
       - `/kaggle/working/LLM-Bias` for code
       - `/kaggle/working/hf_cache` for Hugging Face cache
       - `/kaggle/working/outputs` for results
2. Load code into Kaggle.
    - Option A: Upload this workspace as a Kaggle Dataset and attach it.
    - Option B: Clone your Git repository directly in the notebook.
      - Set project and cache paths before running data/model downloads:
         ```python
         import os
         import sys

         PROJECT_ROOT = '/kaggle/working/LLM-Bias'
         HF_CACHE = '/kaggle/working/hf_cache'

         os.environ['HF_HOME'] = HF_CACHE
         os.environ['HF_DATASETS_CACHE'] = f'{HF_CACHE}/datasets'
         os.environ['TRANSFORMERS_CACHE'] = f'{HF_CACHE}/transformers'

         sys.path.append(f'{PROJECT_ROOT}/src')
         ```
    - Ensure the Python path includes `src`:
       - `import sys; sys.path.append('/kaggle/working/LLM-Bias/src')`
3. Install dependencies.
    - Run:
       ```bash
       pip install -r /kaggle/working/LLM-Bias/requirements.txt
       ```
    - If tokenizer/model loading fails for a specific model family, install optional packages:
       ```bash
       pip install sentencepiece protobuf
       ```
4. Verify GPU and runtime.
    - Quick check:
       ```python
       import torch
       print('cuda_available=', torch.cuda.is_available())
       print('gpu_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
       ```
    - For this experiment, full runs should use GPU. CPU is for smoke tests only.
5. Download and load training data.
    - Built-in loaders in this code support Hugging Face access for:
       - `sst2` via `glue/sst2`
       - `gsm8k` via `gsm8k/main`
       - `squad_v2` via `squad_v2`
    - For `searchqa` and `cm17k`, provide a local JSONL file path with `local_dataset_path` in config.
    - The pipeline constructs grouped permutation data per paper protocol:
       - each original sample becomes 8 order variants
       - each set of variants shares one group ID
6. Load LLM model.
    - Default config uses `Qwen/Qwen2.5-0.5B-Instruct` for free-GPU practicality.
    - For closer fidelity, replace with larger models only if memory permits.
    - Model and tokenizer are loaded through `AutoModelForCausalLM` and `AutoTokenizer`.
7. Configure and run experiment.
    - Use provided config file:
       - `configs/kaggle_sst2_dgao.json`
    - Run DGAO:
       ```bash
       python -m dgao_exp.run_experiment --config /kaggle/working/LLM-Bias/configs/kaggle_sst2_dgao.json
       ```
    - Run SFT baseline:
       ```bash
       python -m dgao_exp.run_experiment --config /kaggle/working/LLM-Bias/configs/kaggle_sst2_dgao.json --mode sft
       ```
    - Run PAFT baseline:
       ```bash
       python -m dgao_exp.run_experiment --config /kaggle/working/LLM-Bias/configs/kaggle_sst2_dgao.json --mode paft
       ```
8. Inspect outputs and metrics.
    - Output JSON is written to `output_dir/results.json`.
    - Key metrics:
       - `accuracy`
       - `consistency_rate`
       - `overconfidence_rate`
    - Compare trend direction against paper findings:
       - DGAO should generally improve accuracy and reduce overconfidence compared with PAFT.
9. Scale up safely.
    - Start with lower `train_groups` and `dgao_steps` to verify pipeline.
    - Increase groups, epochs, and seeds gradually after successful smoke run.
    - Keep `num_order_variants=8` to stay aligned with the paper protocol.
10. Reproducibility and logging checklist.
    - Fix `seed` in config.
    - Save config used for each run.
    - Record GPU type, runtime duration, and library versions.
    - Keep outputs separated per run (for example by folder naming with timestamp or mode).

**Steps**
1. Reconstruct the experiment spec from the paper and lock the reproduction scope.
   - Use the main setup from Section 4.1 and Appendix B as the source of truth.
   - Start with one base model and one dataset family for a minimal end-to-end reproduction, then expand to the full reported suite.
   - Treat the appendix generalization and larger-model runs as optional follow-up scope, not a blocker for the first reproduction pass.
2. Build the dataset construction pipeline.
   - Implement the paper’s sampling and permutation logic for the five datasets used in the experiments: GSM8K, CM17K, SST2, SQuAD v2, and SearchQA.
   - Match the paper’s reported group construction: 2K original samples per dataset, 8 random order variants per sample, yielding 16K augmented training examples per task.
   - Encode the task-specific prompt layouts: 4 QA pairs for math, 16 QA pairs for sentiment, and 5 documents for RAG with one relevant document.
   - Recreate the test-set construction the same way, so evaluation is permutation-aware rather than single-order only.
3. Implement the baseline training paths.
   - Add standard SFT on the original 2K samples.
   - Add PAFT on the 16K augmented samples with full-parameter fine-tuning.
   - Use the same optimizer and schedule settings reported in the paper: batch size 8, learning rate 2e-5, 3 epochs, Adam betas 0.9 and 0.999, epsilon 1e-8.
4. Implement DGAO training.
   - Reproduce the warm-start stage by SFT on half of the augmented data before RL training.
   - Implement the group-based reward computation with inter-group advantage, intra-group advantage, and the weighted hybrid advantage.
   - Use the paper’s reported DGAO hyperparameters: alpha 0.5, beta 0.05, clip epsilon 0.2, random order count 8, and advantage normalization epsilon 1e-8.
   - Keep the reward function rule-based initially: correct answer gets 1, incorrect gets 0.
5. Add the comparison methods used in the paper.
   - Include PPO and GRPO as RL baselines.
   - Add the non-training order-sensitivity baselines the paper compares against when feasible: Prompt, Lost in the middle, PINE, SPHS, GlobalE, and DEmO.
   - Prioritize the methods that are easiest to reproduce with public implementations, then document any missing baselines separately if exact replication is not practical.
6. Implement evaluation and reporting.
   - Calculate Accuracy, Consistency Rate, and Overconfidence Rate exactly as defined in the paper.
   - Run each evaluation with greedy decoding and average results across 3 independent runs.
   - Produce tables that mirror the paper’s main result tables, plus an ablation table for removing inter-group or intra-group advantage.
7. Validate the reproduction.
   - First verify the data pipeline by checking that each original sample produces 8 distinct order variants and that labels remain consistent across variants.
   - Then verify a small training slice can run end-to-end for SFT and DGAO without shape or reward errors.
   - Finally compare the reproduced metrics against the paper’s reported trends: DGAO should improve Accuracy and reduce Overconfidence Rate relative to SFT/PAFT in most tasks, while PAFT should improve consistency at the cost of accuracy.
8. Expand coverage if the first pass is successful.
   - Reproduce the paper’s ablation study without A_group and without A_ind.
   - Reproduce the Appendix C generalization tests on 32-shot and 64-shot SQuAD v2 if the required context length and compute budget are available.
   - Reproduce the larger-model appendix runs only if the smaller-model reproduction is already stable.

**Relevant files**
- No workspace code files are identified yet. The first implementation pass will likely need a new experiment runner, dataset-construction module, training configs, and evaluation scripts once the target project structure is known.
- Use the paper and appendix text as the reference spec for any future local files.

**Verification**
1. Confirm the constructed datasets match the paper’s counts, group sizes, and permutation counts for all five tasks.
2. Run a minimal end-to-end smoke test for one task and one model to confirm SFT, PAFT, and DGAO all execute with the expected input shapes and reward signals.
3. Compare the reproduced metrics against the paper’s directionality, then against the published values where the same base model and task are available.
4. If code is added, run the project’s narrow unit or integration tests around data construction, reward computation, and metric calculation.

**Decisions**
- Scope for the first pass is the main experimental setup in Section 4 and Appendix B, not the full appendix-scale reproduction.
- The plan assumes open-source model weights and public dataset access are available.
- Exact numeric matching may require small implementation choices not fully specified in the paper, so the first goal is trend and protocol fidelity, then numeric closeness.

**Functional Requirements**
1. The pipeline must construct grouped permutation data for each selected dataset, with each original sample mapped to 8 order variants and one stable group identifier.
2. The pipeline must support training modes for SFT, PAFT, and DGAO, and allow reproducible execution through configuration rather than hard-coded constants.
3. The DGAO implementation must compute rule-based rewards, inter-group advantage, intra-group advantage, hybrid advantage, and normalized final advantage during each optimization step.
4. The system must support at least one RL baseline (PPO or GRPO), and should support both for method comparison.
5. The evaluation stage must compute Accuracy, Consistency Rate, and Overconfidence Rate exactly per paper definitions and output task-level summaries.
6. The run workflow must support multi-run evaluation (3 independent runs) and aggregate mean results for each model-task-method combination.
7. The reporting step must produce machine-readable and human-readable outputs (for example JSON or CSV plus markdown tables) that can be compared with the paper tables.
8. The experiment code must expose controls for model choice, task choice, random seed, and training stage selection (SFT warm-start only, full DGAO, or baseline-only runs).

**Non-Functional Requirements**
1. Reproducibility: identical config and seed values must produce the same dataset grouping and statistically close training outcomes across reruns.
2. Traceability: every run must persist metadata including git commit (if present), config snapshot, seed, dataset split version, model checkpoint, and timestamp.
3. Reliability: training and evaluation jobs must fail fast with explicit error messages when data integrity checks or metric assumptions are violated.
4. Scalability: the data and training pipeline should handle all five tasks and multiple model sizes without code changes, using config-only adjustments.
5. Performance: data preprocessing should avoid repeated work via caching or persisted intermediate artifacts, and evaluation should support batched inference.
6. Resource awareness: the implementation should allow reduced-scale runs (smaller sample counts, fewer epochs, fewer tasks) for smoke tests on limited hardware.
7. Maintainability: components for data construction, training, rewards, and metrics should be modular so each can be tested and updated independently.
8. Testability: metric computation and group-construction logic should be covered by automated tests with known expected outputs.
9. Observability: runs should log key progress signals (epoch, loss terms, reward statistics, and metric snapshots) so regressions can be diagnosed quickly.
10. Portability: environment setup should be scripted and version-pinned (Python, torch, transformers, CUDA compatibility) so runs are reproducible across machines.

**Further Considerations**
1. If you want, the next version of the plan can be narrowed to a single target such as only Llama-3.2-3B on SQuAD v2 and SearchQA for a faster reproduction cycle.
2. If strict fidelity matters, the next pass should capture exact prompt templates, answer normalization, and dataset filtering rules from the authors’ code repository when available.
