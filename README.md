# PaperScout

**PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization**

[Project Page](https://agentr1.github.io/paperscout/) | [Paper](https://arxiv.org/pdf/2601.10029) | Accepted to Findings of EMNLP 2026

PaperScout is an autonomous LLM-based agent for academic paper search. Instead of treating retrieval as a one-shot ranking problem or a fixed search-then-expand workflow, PaperScout formulates paper search as a multi-turn decision-making process: at each step, the agent decides whether, when, and how to call `Search` and `Expand` tools according to the papers retrieved so far.

To train this kind of multi-turn retrieval agent, we introduce **Proximal Sequence Policy Optimization (PSPO)**, a process-aware RL algorithm that optimizes complete agent responses as sequence-level actions and incorporates intermediate retrieval rewards.

## News

- 🎉 PaperScout has been accepted to Findings of EMNLP 2026.
- 🚀 Code is available at <https://github.com/AgentR1/PaperScout>.

## Overview

Academic paper search often requires conditional and exploratory behavior. For example, a user may want papers that apply reinforcement learning to protein folding while excluding Transformer-based architectures. Static semantic matching can return papers that are topically similar but constraint-violating, while fixed workflows cannot flexibly decide when to broaden search, deepen citation expansion, or redirect exploration.

PaperScout addresses this by maintaining a paper pool as its latent search state. At each interaction step, the agent observes a compact summary of the pool and the action history, then emits tool calls:

- `Search(query)`: calls scholarly search APIs to retrieve new candidate papers.
- `Expand(paper)`: follows references from an existing paper to discover related papers.

Retrieved papers are scored for relevance, filtered, and merged into the pool. This lets the search strategy evolve over multiple turns rather than following a predefined retrieval pipeline.

<img width="1659" height="634" alt="PaperScout framework" src="https://github.com/user-attachments/assets/087f7fe6-0cd8-4703-95ec-def587d97f10" />

## Why PSPO?

Standard RL methods are not naturally aligned with multi-turn agentic retrieval:

- Token-level PPO assigns turn-level feedback across many individual tokens, producing noisy credit assignment.
- Outcome-oriented methods such as GRPO optimize mostly from final outcomes and do not naturally use intermediate retrieval gains.

PSPO instead treats each complete agent response as the atomic action. It estimates advantages at the sequence level and uses process rewards from intermediate search steps through a learned critic. This matches the granularity of agent-environment interaction and improves training stability.

<img width="1782" height="658" alt="PSPO sequence-level optimization" src="https://github.com/user-attachments/assets/1cf52ad6-5497-4b07-86aa-733b3722af76" />

## Results

We evaluate PaperScout on RealScholarQuery and AutoScholarQuery. In the camera-ready version, PaperScout is trained from Qwen3-4B with PSPO and compared against single-shot retrieval, fixed/multi-turn academic search systems, untrained Qwen3-4B, and a larger untrained Qwen3-Max backbone.

On RealScholarQuery, PaperScout reaches the best post-threshold recall, F1, Recall@k, and LLM-score:

| Model | Precision | F1 | Recall | Recall@all | Recall@100 | Recall@50 | Recall@25 | LLM-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaSa | 0.415 | 0.417 | 0.541 | 0.598 | 0.591 | 0.572 | 0.517 | 2.111 |
| SPAR | 0.412 | 0.408 | 0.496 | 0.545 | 0.545 | 0.537 | 0.504 | 2.415 |
| PaperScout-Qwen3-4B | 0.404 | 0.411 | 0.497 | 0.545 | 0.543 | 0.535 | 0.498 | 2.261 |
| PaperScout-Qwen3-Max | 0.435 | 0.427 | 0.562 | 0.664 | 0.648 | 0.606 | 0.532 | 2.483 |
| **PaperScout** | **0.442** | **0.441** | **0.574** | **0.691** | **0.683** | **0.646** | **0.577** | **2.576** |

On AutoScholarQuery, PaperScout also achieves the strongest overall performance:

| Model | Precision | F1 | Recall | Recall@all | Recall@100 | Recall@50 | Recall@25 | LLM-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaSa | 0.095 | 0.129 | 0.442 | 0.699 | 0.579 | 0.463 | 0.344 | 2.186 |
| SPAR | 0.091 | 0.131 | 0.386 | 0.561 | 0.535 | 0.466 | 0.341 | 2.295 |
| PaperScout-Qwen3-4B | 0.092 | 0.128 | 0.382 | 0.564 | 0.540 | 0.448 | 0.324 | 2.202 |
| PaperScout-Qwen3-Max | 0.102 | 0.115 | 0.427 | 0.770 | 0.584 | 0.469 | 0.336 | 2.368 |
| **PaperScout** | **0.115** | **0.134** | **0.459** | **0.811** | **0.636** | **0.496** | **0.357** | **2.467** |

PSPO also outperforms PPO and GSPO on RealScholarQuery:

| Method | Precision | F1 | Recall | LLM-score |
| --- | ---: | ---: | ---: | ---: |
| PPO | 0.405 | 0.408 | 0.537 | 2.417 |
| GSPO | 0.433 | 0.439 | 0.557 | 2.510 |
| **PSPO** | **0.442** | **0.441** | **0.574** | **2.576** |

## Repository Structure

```text
PaperScout/
|-- arft/                    # Agent RL training components and PSPO-related trainer code
|-- recipe/paper_search/     # Paper search environment, prompts, tools, and agent flow
|-- examples/                # Example training scripts
`-- verl/                    # verl-based RL infrastructure
```

Important files:

- `recipe/paper_search/base.yaml`: PaperScout agent-flow configuration.
- `recipe/paper_search/paper_search_agent_flow.py`: multi-turn paper search loop with `Search` and `Expand` tool execution.
- `arft/main_agent_ppo.py`: training entry point.
- `examples/run_qwen3-4b-mlflow-dbs.sh`: example Qwen3-4B training script.

## Quick Start

Clone the repository:

```bash
git clone git@github.com:AgentR1/PaperScout.git
cd PaperScout
```

Install dependencies following the verl environment requirements used by this repository:

```bash
cd verl
pip install -r requirements.txt
pip install -r requirements_sglang.txt
cd ..
```

Prepare the training and validation data paths used by the example script:

```text
$HOME/data/pasa/train.parquet
$HOME/data/pasa/test.parquet
```

Then update the model and critic paths in `examples/run_qwen3-4b-mlflow-dbs.sh` to match your local environment.

Run training:

```bash
bash examples/run_qwen3-4b-mlflow-dbs.sh
```

The default agent-flow config uses up to 5 interaction steps and up to 5 parallel tool calls per step:

```yaml
- name: paper_search_agent
  _target_: recipe.paper_search.paper_search_agent_flow.PaperSearchAgentFlow
  max_steps: 5
  max_parallel_calls: 5
```

## Citation

If you are interested in our work, please don't hesitate to contact us for discussion and collaboration!

```bibtex
@inproceedings{pan2026paperscout,
  title = {PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization},
  author = {Pan, Tingyue and Ouyang, Jie and Cheng, Mingyue and Li, Qingchuan and Liu, Zirui and Wang, Daoyu and Pan, Mingfan and Yu, Shuo and Liu, Qi and Chen, Enhong},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year = {2026}
}
```
