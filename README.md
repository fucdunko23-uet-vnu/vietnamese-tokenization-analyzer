# 🇻🇳 Vietnamese Tokenization & Embedding Representation Analyzer

A focused exploration into Vietnamese NLP, tokenization behavior, and multilingual sentence embeddings (`intfloat/multilingual-e5-small`). This mini-project demonstrates the mathematical properties and core limitations of naive vector aggregation (Bag-of-Words summation) vs. modern contextual transformer architectures.

---

## 💡 Overview & Key Insight

When building Large Language Models (LLMs) and NLP pipelines, understanding how word embeddings represent semantics and word order is fundamental. 

This project explores a classic NLP challenge using Vietnamese sentence pairs:
* **Sentence 1**: `["chó", "cắn", "người"]` *(Dog bites man)*
* **Sentence 2**: `["người", "cắn", "chó"]` *(Man bites dog)*

### The Core Finding
By aggregating token embeddings via unweighted summation without positional encodings:

$$\text{Representation} = \sum_{i=1}^{N} \text{Embedding}(w_i)$$

The model produces **mathematically identical representation vectors** for both sentences despite their completely opposite semantic meanings.

```python
# Output verification from mini_project_3.py
np.allclose(rep_1, rep_2) # Returns True!
```

### Why This Matters for LLMs
- **Bag-of-Words Limitation**: Pure vector summation loses word order and syntactic structure.
- **Need for Positional Embeddings**: Highlights why modern Transformer architectures (BERT, GPT, LLaMA) require **Positional Encodings** (e.g., Absolute, Relative, or Rotary Position Embeddings - RoPE) to distinguish sequence context.

---

## 🛠️ Tech Stack & Model

- **Language**: Python 3.10+
- **Framework**: `sentence-transformers`, `numpy`
- **Pretrained Model**: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (384-dimensional multilingual dense embeddings)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/fucdunko23-uet-vnu/vietnamese-tokenization-analyzer.git
cd vietnamese-tokenization-analyzer
```

### 2. Install Dependencies
```bash
pip install sentence-transformers numpy
```

### 3. Run the Analyzer
```bash
python mini_project_3.py
```

---

## 📂 Project Structure

```
vietnamese-tokenization-analyzer/
├── mini_project_3.py   # Main script demonstrating embedding sum & order insensitivity
├── README.md           # Project documentation
└── .gitignore          # Git ignore file
```

---

## 🎓 Learning Objectives & Takeaways

1. **Embedding Extraction**: Extracting dense 384-dimensional feature vectors using multilingual transformer encoders.
2. **Vector Space Operations**: Analyzing vector addition and semantic invariant properties.
3. **Transformer Foundations**: Building foundational intuition for self-attention mechanisms and positional encoding requirements in LLM design.

---

## 👤 Author

**fucdunko23-uet-vnu**  
*GenAI & LLM Architecture Learning Roadmap*
