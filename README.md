# LLM-Imputation-Pancreatic-Cancer

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: AIM](https://img.shields.io/badge/Paper-Artificial%20Intelligence%20in%20Medicine-green.svg)]()

**A Hierarchical LLM-Augmented Framework for Missing Data Imputation in Pancreatic Cancer Electronic Health Records**

</div>

## 📝 Paper Information

- **Title**: A Hierarchical LLM-Augmented Framework for Missing Data Imputation in Pancreatic Cancer Electronic Health Records: Validation Through Downstream Prognostic Analysis
- **Journal**: Artificial Intelligence in Medicine (IF: 7.4, JCR Q1)
- **Status**: Under Review
- **Word Count**: ~6,200 words

## 🎯 Research Highlights

- ✅ **Novel Framework**: First study to systematically apply LLMs for EMR missing data imputation
- ✅ **Hierarchical Design**: Three-tier approach (Rule-based + LLM inference + Deep reasoning)
- ✅ **Clinical Validation**: C-index improvement from 0.61 to 0.72 (0.11 gain)
- ✅ **External Validation**: Consistent with SEER database and published literature
- ✅ **Interpretability**: Confidence scores for each imputation

## 📊 Key Results

| Metric | Pre-Imputation | Post-Tier 1 | Post-Tier 2 | Improvement |
|--------|---------------|-------------|-------------|-------------|
| **C-index** | 0.61 | 0.68 | **0.72** | +0.11 |
| **12-month AUC** | 0.64 | 0.68 | **0.71** | +0.07 |
| **Brier Score** | 0.24 | 0.21 | **0.18** | -0.06 |

## 📁 Repository Structure

```
.
├── data/
│   ├── raw/                    # Original EMR data (de-identified)
│   ├── processed/              # Processed datasets
│   └── imputed/                # Tier 1 & Tier 2 imputed data
├── code/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── missing_analysis.py     # Missing pattern analysis
│   ├── tier1_imputation.py     # Rule-based imputation
│   ├── tier2_llm_imputation.py # LLM-based imputation
│   ├── survival_analysis.py    # Cox regression and validation
│   └── generate_figures.py     # Figure generation scripts
├── figures/
│   ├── Figure1_框架架构图.png
│   ├── Figure2_缺失模式热力图.png
│   ├── Figure3_LLM置信度分布.png
│   └── Figure4_外部验证森林图.png
├── paper/
│   ├── 完整投稿论文_AIM.html    # Word-compatible manuscript
│   ├── manuscript.tex          # LaTeX source
│   └── AIM_投稿指南.md         # Submission guidelines
├── skill/
│   └── medical-data-research/  # Custom OpenClaw skill
├── README.md
└── LICENSE
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/anitafang_create/LLM-Imputation-Pancreatic-Cancer.git
cd LLM-Imputation-Pancreatic-Cancer

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
# Load and preprocess data
from code.data_loader import load_medical_data
df = load_medical_data('data/raw/pancreatic_cancer.csv')

# Run missing analysis
from code.missing_analysis import analyze_missing_comprehensive
results = analyze_missing_comprehensive(df)

# Tier 1: Rule-based imputation
from code.tier1_imputation import apply_rule_based_imputation
df_tier1 = apply_rule_based_imputation(df)

# Tier 2: LLM imputation
from code.tier2_llm_imputation import apply_llm_imputation
df_tier2 = apply_llm_imputation(df_tier1)

# Survival analysis
from code.survival_analysis import evaluate_survival_models
c_index, auc = evaluate_survival_models(df_tier2)
```

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{anonymous2024llm,
  title={A Hierarchical LLM-Augmented Framework for Missing Data Imputation in Pancreatic Cancer Electronic Health Records: Validation Through Downstream Prognostic Analysis},
  journal={Artificial Intelligence in Medicine},
  year={2024},
  publisher={Elsevier}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Clinical data provided by [Hospital Name] (anonymized)
- LLM inference powered by OpenAI GPT-4
- Survival analysis using [Lifelines](https://lifelines.readthedocs.io/)

## 📞 Contact

For questions or collaboration inquiries, please open an issue or contact the corresponding author.

---

<div align="center">

**📝 Paper Status**: Submitted to Artificial Intelligence in Medicine  
**🔄 Last Updated**: 2026-03-22  
**🌐 GitHub**: https://github.com/anitafang_create/LLM-Imputation-Pancreatic-Cancer

</div>
