Note: This is an alpha release to prove a HMoE concept.

# Hierarchical Mixture-of-Experts (MoE) Pipeline

**Status:** Under Active Development | **Focus:** REVERSAL DETECTION | **Temporary Repository for validation**

This branch introduces a complete overhaul of the RPulsar machine learning backend. After extensive research and prototyping, we are transitioning from a monolithic dense network architecture to a high-performance **Hierarchical Mixture-of-Experts (MoE)** approach powered by **Microsoft Tutel**. 

Example what you can build with this, with some effort (untuned example): 

![Example](docs/this-is-what-you-can-build-with-it-untuned-example.png)

The above example is OOS and untuned. Given the limited feature-set the results are very promising.

At the pinnacle of this architecture sits a deterministic macro-router that fundamentally strips the neural network of high-level guesswork. Instead of allowing a traditional, black-box gating network to estimate the overarching market environment, this root node acts as an uncompromising mathematical gatekeeper driven by a strictly defined moving average regime filter. By evaluating whether the market is in a validated uptrend, downtrend, or untradeable chop, the router dynamically masks the tensor pathways in real-time. This ensures that entire sub-trees of the model are hard-pruned from the computational graph the moment their specific market conditions are absent, instantly halting capital deployment during sideways chop and preserving both processing power and gradient integrity.

Beneath this macro-routing layer lies an asymmetric, hierarchical deployment of specialized expert squads, intentionally unbalanced to reflect the divergent realities of bull and bear markets. When the root node detects a bullish regime, it activates a neural sub-router exclusively fed with bottom-finding candlestick heuristics and momentum oscillators tailored to accurately buy the dip. Conversely, a bearish regime triggers a completely separate squad of experts trained explicitly on top-reversal patterns to aggressively short the rip. This intentional asymmetry is crucial; it prevents the network from wasting parameter capacity on bullish continuation patterns during a market crash, forcing the active experts to become hyper-specialized snipers for their designated market phase.

The true elegance of this system is realized in the data pipeline and loss calculation, which seamlessly bridge the deterministic routing with the neural backpropagation. The pipeline dynamically synthesizes a unified target tensor on the fly, feeding bottom-labels to the loss function during uptrends and top-labels during downtrends. Because the root router only allows the regime-appropriate experts to generate logits at any given timestep, the PyTorch loss function naturally penalizes and rewards the correct subnetworks without requiring the neural architecture to explicitly comprehend the shifting targets. This creates a beautifully isolated training environment where quantitative macro-rules dictate the battleground, and deep learning is left to purely optimize the tactical execution.

The problems that "are solved" (depends on your features stack):

- Extreme class imbalance (sparsity): Handled via targeted focal loss and by isolating the "hunting grounds" so the network isn't drowned in endless zeros.

- Monolithic feature processing (The "Muddle"): Solved by ensuring each indicator class (RSI, Candlesticks, MACD) has its own dedicated "brains" (Expert Squads) to learn domain-specific patterns before fusion.

- Neural networks guessing macro trends: Solved by decoupling trend identification from pattern recognition, using static routing via a deterministic macro regime filter to dictate the battleground.

- Capital bleed in sideways markets (The "Chop"): Solved by the root router's 0 state, which instantly masks out non-trending regimes and prevents the network from overtrading unpredictable noise.

- Contextual feature contradiction: Solved through Asymmetric Expert Isolation. By stripping bullish signals (like Hammers) from the Bear Squad and vice versa, the network never wastes parameter capacity evaluating noise that conflicts with the macro trend.

- Dynamic target ambiguity: Solved by on-the-fly target synthesis. The pipeline dynamically swaps the ground-truth labels (feeding "bottoms" during uptrends and "tops" during downtrends) seamlessly to the loss function based on the active regime.

- Look-ahead bias (Data Leakage): Solved mathematically at the base layer utilizing strictly causal, asymmetrically left-padded Temporal Convolutional Networks (TCNs). Future price action is physically blocked from contaminating the current state tensor.

- Computational/Parameter bloat: Solved by sparse activation. Because of the MoE gating and Depthwise Separable Convolutions, only a highly specialized fraction of the network's parameters are calculated and updated at any given timestep, saving massive VRAM and compute.

- The "Black-Box" dilemma: Solved by the integrated chunked parallel occlusion pipeline. The architecture doesn't just give you a confidence score; it mathematically attributes that score back to the exact features (e.g., RSI or a specific Doji) that caused the experts to fire.

Note: a general rule is "garbage in, garbage out". You need to make superb features in order to make it reliable and make it perform well. The engine should be solid. I am developing on top of it and it gets better and better with feature improvements. When trading with this structure, implement proper guardrails. Eg cover for macro shocks.

✅ With well-engineered features, this setup could detect reversals reliably for:

- Short-term sequences (intraday or minute-level)
- Structured features (indicators, returns, momentum)

⚠️ Reliability drops if:

- Features are sparse or noisy
- Sequence length does not capture enough context (configurable)

---

## Who is this for?

This is **not** a beginner-friendly "click and run" trading bot.  
It is aimed at:
- Experienced retail / semi-pro quants who already have a solid data pipeline
- Developers comfortable with PyTorch, CUDA, and MoE architectures
- Researchers interested in sparse event detection and regime-aware modeling

If you're new to ML or algorithmic trading, this repository will likely be overwhelming.

---

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 3070 or better recommended)
- WSL2 (Ubuntu) environment strongly recommended
- Working installation of `bp.markets.ingest` with HTTP API running on port 8000
- Proper pivot-finder labels configured in your data pipeline (smear labels, eg 4 after pivot)
- Basic familiarity with PyTorch and MoE concepts

---

## 🎯 The Goal: A Surgical Pipeline

Standard dense neural networks struggle in financial markets because they are forced to process every market regime using the same shared weights. A model trying to learn the geometric structure of a V-bottom gets "lobotomized" when forced to simultaneously process a sparse, high-impact macro shock like an NFP release.

**The Solution:** We are building a pipeline where each sub-model (an "Expert") is encapsulated and optimized strictly for its specific signal-to-noise ratio. The Tutel Gateway acts as the orchestrator, dynamically routing our 24-candle H4 frames (later other TF, H4 remains the focus because of performance) to the exact expert qualified to interpret the current market regime.

**Previously:** Previously we had a hybrid of Macro Stencil with RNN. While this model works pretty well, I am using it atm, the problem remains that it is fairly complex because it combines multiple approaches in a single model/engine. It's easy to introduce bugs. We are now going to split up the models and each of the models will get its specific "knowledge domain".

---

## 🧠 The Expert Modules

Instead of one massive model, we will train multiple specialized, lightweight models. Each expert owns a specific domain of the market data.

Models can be routed using neural nets or deterministic rules.

---

## ⚙️ Core Infrastructure: Microsoft Tutel

To make this computationally feasible for retail cloners and researchers without requiring enterprise data centers, we use **Microsoft Tutel**.

* **Dynamic Gating:** Tutel provides high-performance CUDA kernels that handle the "All-to-All" communication. It evaluates the gating features (e.g., current volatility, time-to-event) and routes the 24-candle token strictly to the Top-K experts.
* **Sparsity Handled:** By utilizing "Expert Dropout" and a Shared "Null" Expert for sideways noise, we prevent specialized models (like the Macro Expert) from hallucinating during quiet market hours.
* **Mechanical Sympathy:** The MoE allows us to scale the intellectual capacity of the bot massively without a linear increase in VRAM or inference latency. 

---

## 🛠️ Developer UX & Setup

This upgrade embraces Infrastructure-as-Code. The setup process for cloners is being streamlined into a single execution script (`setup.sh`) that automatically handles CMake upgrades, specific Clang compiler requirements, and native CUDA architecture targeting to ensure Tutel compiles flawlessly on your specific silicon.

## Quick Setup

### 📦 1. Installation

**Clone the Repository:**
```bash
git clone https://github.com/jpueberbach4/bp.markets.ml-alpha.git
cd bp.markets.ml
```

**Environment Setup:**

We recommend using Python 3.9+ and a virtual environment.

```bash
./setup.sh
```

After, for each WSL window you start, you will need to (re)activate the venv environment

```sh
source ./venv/bin/activate
```

## ⚙️ 2. Configuration Overrides

The system is designed with a Base-Override architecture. Do not modify config.py directly. Instead, create a config.user.py to store your private "Champion" parameters.

Create config.user.py (from config.py):

```bash
cp config.py config.user.py
```

Put any overrides in config.user.py. You need to put overrides in there since the configuration is not optimal. See below.

**Note: Remove the config.user include block from the bottom of the file, in case you have copied from config.py.**

### 3. Data

This engine uses the HTTP API of the `bp.markets.ingest` project. Make sure the data for your asset is available and the API service is running on port 8000.

## 4. Configuration

(I cannot share my config/strategy. If you want to use this, you will need to write your own data pipeline. There is an untuned example config available)

## 5. Initial execution

```bash
./train.sh
```

This will perform some steps:

- Train
- Verify gating and expert distribution
- Plots the example bottom/tops detects to a web-window

## 5. Future

- Multihead attention will be added to get multiple outputs (eg vola info, sentiment info, ...)
- Will be used for live-trading

# LICENSE?

MIT. Do as you please with it.
