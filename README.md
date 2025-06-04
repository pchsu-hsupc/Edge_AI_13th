# Edge_AI_13th

**Model**: [LLaMA-3.2B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  
**Dataset**: [WikiText-2 (raw)](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-2-raw-v1)  
**Platform**: NVIDIA T4 (16GB GPU memory) with CUDA 12.2 \
**Python Version**: 3.10.12

---

## 🔧 Environment Setup

Install necessary packages:

```bash
sudo apt update
sudo apt install -y build-essential python3-venv python3-pip git
```

Create and activate a virtual environment:

```bash
python3 -m venv <your_env>
source <your_env>/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
## 📊 Evaluation Criteria
To evaluate the model (throughput and perplexity):

```bash
python result.py
```
