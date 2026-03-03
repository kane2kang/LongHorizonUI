<div align="center">

# 🖥️ LongHorizonUI

**A Long-Horizon GUI Automation Agent Framework with Enhanced Perception, Deep Reflection, and Compensating Execution**

<p>
  <a href="https://huggingface.co/datasets/KaneKang/LonghorizonUI"><img src="https://img.shields.io/badge/🤗 Dataset-LongGUIBench-yellow" alt="Dataset"></a>
  <a href="https://openreview.net/pdf?id=BK7Mk5d4WE"><img src="https://img.shields.io/badge/Paper-OpenReview-red" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"></a>
</p>

</div>

---

## 📖 Introduction

LongHorizonUI is an agent framework designed for **long-horizon GUI automation tasks**. Existing GUI agents suffer from rapid success-rate degradation in long-step tasks (>10 steps) due to error accumulation. LongHorizonUI addresses this problem through three core modules:

| Module | Description |
|--------|-------------|
| **Multi-source Enhanced Perceiver (MEP)** | Runs icon detection and OCR in parallel, resolves compound widget ambiguity via IoU semantic binding, and repairs missing key elements with template matching |
| **Deep Reflective Decider (DRD)** | Multi-step look-ahead reasoning, retrospective action review, and causal inference on UI states for high-quality action decisions |
| **Compensating Action Executor (CAE)** | Three-level fallback strategy (Index → Relative → Absolute+ε), post-execution verification, progress monitoring, and automatic rollback |

<p align="center">
  <img src="https://media.githubusercontent.com/media/kane2kang/LongHorizonUI/main/page/figure/overview.png" width="90%" alt="Framework Overview">
</p>



---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/kane2kang/LongHorizonUI.git
cd LongHorizonUI

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM API

```bash
cp .env.example .env
# Edit the .env file and fill in the API keys for your LLM provider
```

**Supported LLM Providers:**

| Provider | Required Environment Variables |
|----------|-------------------------------|
| Google Gemini (Recommended) | `GOOGLE_PROJECT`, `GOOGLE_LOCATION` |
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| OpenAI | `OPENAI_ENDPOINT`, `OPENAI_API_KEY` |

### 3. Download Dataset (Optional)

We provide the **LongGUIBench** benchmark dataset for evaluation:

🤗 **[LongGUIBench on Hugging Face](https://huggingface.co/datasets/KaneKang/LonghorizonUI)**

After downloading, place the data under the `data/` directory:

```
data/
├── general/          # General application scenarios
│   ├── app_a/
│   │   ├── task_001/
│   │   │   ├── screenshot/     # UI screenshot sequences
│   │   │   │   ├── 001.png
│   │   │   │   ├── 002.png
│   │   │   │   └── ...
│   │   │   └── task_infos.json # Task description and annotations
│   │   └── ...
│   ├── app_b/
│   └── ...
└── game/             # Game application scenarios
    ├── hero/
    └── ...
```

**`task_infos.json` format example:**

```json
{
  "task_name": "Create a new email in a mail app and send it to a contact",
  "task_steps": [
    {"action": "Click the menu button in the top-left corner"},
    {"action": "Select the compose email option"},
    {"action": "Enter the recipient address"},
    {"action": "Enter the email subject"},
    {"action": "Click the send button"}
  ]
}
```

---

## 💻 Usage

### Mode 1: Offline (Screenshot Simulation)

**No phone connection required.** Simulates the agent's full reasoning and execution pipeline based on pre-recorded screenshot sequences. Suitable for:

- Offline evaluation and experiment reproduction
- Development and debugging without an Android device

```bash
# Low instruction mode (detailed step-by-step instructions provided)
python run.py offline \
  --data_dir data/general/app_a \
  --instruction_level low \
  --provider gemini \
  --model gemini-2.5-pro

# High instruction mode (only task description provided, agent plans autonomously)
python run.py offline \
  --data_dir data/game/game_a \
  --instruction_level high \
  --provider gemini \
  --model gemini-2.5-pro
```

### Mode 2: Live (USB-Connected Device Execution)

Connects to a real Android device via USB. The agent captures screenshots, perceives, decides, and controls the device in real time.

**Prerequisites:**
- USB debugging is enabled on the Android device
- `adb devices` can detect the device

```bash
# Interactive task input
python run.py live \
  --provider gemini \
  --model gemini-2.5-pro

# Load task from file
python run.py live \
  --task_file data/general/app_a/task_001/task_infos.json \
  --provider gemini \
  --model gemini-2.5-pro

# Specify task directly (High mode)
python run.py live \
  --task "Open Settings and connect to a WiFi network" \
  --instruction_level high \
  --provider gemini \
  --model gemini-2.5-pro
```

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--provider` | LLM provider | `gemini` |
| `--model` | Model name | `gemini-2.5-pro` |
| `--instruction_level` | Instruction level: `high` / `low` | `low` |
| `--max_steps` | Maximum execution steps | `100` |
| `--temperature` | LLM sampling temperature | `0.4` |
| `--output_dir` | Output directory | `./output` |


---

## 📊 Main Results

### LongGUIBench

On our self-constructed LongGUIBench, LongHorizonUI significantly outperforms existing methods across both general and game long-horizon scenarios.

<p align="center">
  <img src="https://media.githubusercontent.com/media/kane2kang/LongHorizonUI/main/page/figure/longbench.png" width="85%" alt="LongGUIBench Results">
</p>

### ScreenSpot

On the ScreenSpot cross-platform UI grounding benchmark, LongHorizonUI surpasses previous state-of-the-art methods, validating the effectiveness of the IoU semantic binding strategy in the enhanced perception module.

<p align="center">
  <img src="https://media.githubusercontent.com/media/kane2kang/LongHorizonUI/main/page/figure/screenspot.png" width="85%" alt="ScreenSpot Results">
</p>

---


## 📝 Citation

If this project is helpful for your research, please cite:

```bibtex
@inproceedings{anonymous2026longhorizonui,
  title={LongHorizon{UI}: A Unified Framework for Robust long-horizon Task Automation of {GUI} Agent},
  author={Anonymous},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=BK7Mk5d4WE}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.