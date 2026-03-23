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
├── src/                        # Source code
│   ├── missing_analysis.py     # Missing pattern analysis
│   ├── tier1_imputation.py     # Rule-based imputation
│   ├── tier2_llm_imputation.py # LLM-based imputation
│   ├── data_validation.py      # Data validation
│   ├── generate_all_figures.py # Figure generation
│   ├── generate_figure4.py     # External validation figure
│   ├── llm_imputation_strategy.py  # LLM strategy definition
│   └── create_medical_skill.py # OpenClaw skill creator
├── data/                       # Data files
│   └── *.json                  # Configuration and metadata
├── figures/                    # Generated figures
│   ├── Figure1_框架架构图.{png,pdf}
│   ├── Figure2_缺失模式热力图.{png,pdf}
│   ├── Figure3_LLM置信度分布.{png,pdf}
│   ├── Figure4_外部验证森林图.{png,pdf}
│   └── 缺失*.{png,pdf}         # Additional analysis figures
├── docs/                       # Documentation and papers
│   └── 完整论文_胰腺癌EMR数据LLM补全研究.md  # Full manuscript
├── results/                    # Analysis results (output directory)
├── .github/workflows/          # CI/CD workflows
├── README.md                   # This file
└── requirements.txt            # Python dependencies
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
from src.missing_analysis import analyze_missing_comprehensive
df = pd.read_csv('your_data.csv')
results = analyze_missing_comprehensive(df)

# Tier 1: Rule-based imputation
from src.tier1_imputation import apply_tier1_imputation
df_tier1 = apply_tier1_imputation(df)

# Tier 2: LLM imputation
from src.tier2_llm_imputation import apply_tier2_imputation
df_tier2 = apply_tier2_imputation(df_tier1)

# Data validation
from src.data_validation import validate_data
checks_passed, checks_failed = validate_data(df_tier2)

# Generate figures
from src.generate_all_figures import generate_all_figures
generate_all_figures(df_tier2, output_dir='figures/')
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
