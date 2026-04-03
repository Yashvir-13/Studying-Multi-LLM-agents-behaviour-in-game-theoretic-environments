# Emergent Phase Transitions in Multi-Agent LLM Systems

A research framework for investigating emergent behaviors — cooperation, defection, deception, and trust dynamics — in multi-agent systems driven by Large Language Models. Runs controlled experiments across two game-theoretic environments to detect non-linear regime shifts in agent behavior.

## Repository Structure

```
├── agents/
│   ├── base_agent.py                # LLM agent for Public Goods Game (neutral prompt framing)
│   ├── signaling_agents.py          # LLM Sender & Receiver agents for Signaling Game
│   ├── signaling_baseline_agents.py # Algorithmic baselines (AdaptiveTrust, StrategicLiar, etc.)
│   ├── baseline_agents.py           # Public Goods baselines (TitForTat, Random, etc.)
│   └── llm_client.py               # Ollama API wrapper (JSON mode, think toggle, retries)
├── environments/
│   ├── public_goods.py              # Public Goods Game
│   └── signaling_game.py           # Signaling Game (asymmetric payoffs, 4 topologies)
├── logs/                            # All experiment outputs (auto-organized)
├── config.py                        # ExperimentConfig dataclass (YAML/JSON loading)
├── public_goods_config.yaml         # Public Goods Game config
├── signaling_game_config.yaml       # Signaling Game config
├── main.py                          # CLI entry point (single / sweep / baseline)
├── experiment_runner.py             # Parameter sweep orchestrator with checkpointing
├── simulation_runner.py             # Agent-environment loop (parallel LLM calls)
├── metrics.py                       # Statistical metrics (both environments)
├── visualization.py                 # Plot generation from sweep reports
└── requirements.txt                 # Python dependencies
```

## Getting Started

### Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running (`ollama serve`)
3. **Model** pulled: `ollama pull qwen3.5:9b`

### Installation

```bash
git clone <repo-url>
cd "Emergent Behaviour in Multi LLM agents"
pip install -r requirements.txt
```

## Environments

### 1. Public Goods Game

Agents decide how much of their endowment to contribute to a shared pool each round.

**Payoff**: `Pᵢ = (Endowment − Contributionᵢ) + (Σ Contributions × M) / N`

| Concept | Contribution | Why |
|---------|-------------|-----|
| **Nash Equilibrium** | 0 (defect) | Each token contributed returns only `M/N < 1` to the contributor |
| **Social Optimum** | Full endowment | Group total is maximized when everyone contributes |
| **Research Question** | — | Does memory, scale (N), or information structure shift LLM agents from Nash toward Social Optimum? |

### 2. Signaling Game (Information Asymmetry)

A repeated signaling game studying emergent deception under information asymmetry.

**Rules:**
1. **Nature** privately assigns each Sender an item quality: `HIGH` or `LOW`
2. **Senders** (LLMs) observe their quality and send a message (`"high"` or `"low"`) — they may lie
3. **Receivers** observe messages and decide to `INVEST` or `PASS`
4. **Payoffs** depend on the TRUE state, not the reported message

**Asymmetric Payoff Matrix ("Scammer Incentive"):**

| Scenario | Sender Gets | Receiver Gets |
|----------|-------------|---------------|
| TRUE=HIGH, Receiver INVESTS | +2 | +3 |
| TRUE=LOW, Receiver INVESTS (scam) | **+10** | -1 |
| Receiver PASSES | 0 | 0 |

**Communication Topologies:**

| Topology | Description |
|----------|-------------|
| `full` | Every sender → every receiver |
| `star` | All senders → one central receiver (**requires `num_receivers: [1]`**) |
| `ring` | Each sender → one adjacent receiver (1:1 wraparound) |
| `broadcast` | Only Sender_0 → all receivers (**requires `num_senders: [1]`**) |

**Agent Types:**
- `llm` — Both senders and receivers are LLMs
- `llm_sender_adaptive_receiver` — LLM senders vs. algorithmic Bayesian receivers (recommended for studying deception)

## Configuration

All experiment parameters are defined in YAML config files:

- **`public_goods_config.yaml`** — Public Goods Game
- **`signaling_game_config.yaml`** — Signaling Game

```yaml
# Key settings (signaling_game_config.yaml)
model: "qwen3.5:9b"
think: false                    # Disable reasoning for speed & parse reliability

topology: ["ring", "star"]     # Sweep over topologies
memory_limit: [3, 5]           # Sweep over memory limits

sender_reward_high_invest: 2.0  # Legitimate sale pays 2
sender_reward_low_invest: 10.0  # Scam pays 10 (5x incentive to lie)

agent_type: "llm_sender_adaptive_receiver"
```

## Usage

### 1. Single Run (Debugging)

```bash
python3 main.py --mode single --config public_goods_config.yaml
```

Runs one simulation with the first value from each grid parameter.

### 2. Parameter Sweep (Research)

```bash
# Public Goods sweep
python3 main.py --mode sweep --config public_goods_config.yaml

# Signaling Game sweep
python3 main.py --mode sweep --config signaling_game_config.yaml
```

Runs all combinations from the config grid. Saves checkpoints after each config — resume interrupted sweeps with:

```bash
python3 main.py --mode sweep --config signaling_game_config.yaml --resume logs/baseline_llm_sender_adaptive_receiver_XXXXXX
```

### 3. Baseline Comparison (Public Goods only)

```bash
python3 main.py --mode baseline --config public_goods_config.yaml
```

Runs `AlwaysCooperate`, `AlwaysDefect`, `Random`, and `TitForTat` agents through the same grid.

### 4. Visualization

```bash
python3 visualization.py logs/<sweep_dir>/final_sweep_report.json
```

Auto-detects the environment type and generates appropriate plots.

**Public Goods plots:** cooperation rate, entropy, Gini coefficient, defector ratio, per-trial variance

**Signaling Game plots:** deception rate, trust rate, informed trust, deception success, receiver accuracy

## Metrics

### Public Goods Game

| Metric | Description | Range |
|--------|-------------|-------|
| **Cooperation Rate** | `Σ contributions / (N × endowment)` | 0.0 → 1.0 |
| **Behavioral Entropy** | Shannon entropy of contribution distribution | 0 → log₂(N) |
| **Gini Coefficient** | Inequality in agent rewards | 0 (equal) → 1 |
| **Defector Ratio** | Fraction of agents contributing 0 | 0.0 → 1.0 |
| **Stability Index** | Rolling variance of cooperation rate | Low = stable |
| **Time-to-Convergence** | Round where cooperation stabilizes | Round # or None |

### Signaling Game

| Metric | Description | Range |
|--------|-------------|-------|
| **Deception Rate** | Fraction of messages that differ from true state | 0.0 → 1.0 |
| **Trust Rate** | Fraction of receiver decisions that are INVEST | 0.0 → 1.0 |
| **Informed Trust** | Trust rate excluding receivers with empty inboxes | 0.0 → 1.0 |
| **Deception Success** | Fraction of lies that led to INVEST (topology-aware) | 0.0 → 1.0 |
| **Receiver Accuracy** | Fraction of optimal receiver decisions (topology-aware) | 0.0 → 1.0 |

## Design Decisions

- **Neutral Prompt Framing**: Agents are told they're in an "information exchange experiment," not a "signaling game," to avoid LLM training bias
- **Thinking Disabled**: `think: false` is used with Qwen3.5 for consistent output parsing and ~10x faster inference vs. reasoning mode
- **Asymmetric Payoffs**: The "Scammer Incentive" (LOW+invest = +10) creates a game-theoretic tension between short-term scam profits and long-term reputation
- **Parallel LLM Calls**: All agents in a round query the LLM simultaneously via `ThreadPoolExecutor`
- **Checkpointing**: Sweep progress is saved after every config, enabling resume on interruption
- **Topology-Aware Metrics**: Deception success and receiver accuracy only count agents connected via the communication topology

## Output Structure

Each sweep creates a timestamped directory:

```
logs/baseline_llm_sender_adaptive_receiver_20260326-105655/
├── experiment_config.yaml                          # Full config snapshot
├── final_sweep_report.json                         # Aggregated metrics
├── raw_S2_R2_Tring_Mem3_Aper_sender_F1_*.json     # Raw trial data
├── deception_plot.png
├── trust_plot.png
├── deception_success_plot.png
├── receiver_accuracy_plot.png
└── convergence_summary.txt
```
