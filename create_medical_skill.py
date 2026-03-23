#!/usr/bin/env python3
"""
创建 medical-data-research skill
用于医学数据处理、AI算法设计和论文生成
"""

import os
from pathlib import Path

# Skill 配置
SKILL_NAME = "medical-data-research"
SKILL_PATH = "/Users/fanganita/.openclaw/workspace/skills/" + SKILL_NAME

print("="*80)
print(f"🛠️ 创建 Skill: {SKILL_NAME}")
print("="*80)

# 创建目录结构
os.makedirs(f"{SKILL_PATH}/scripts", exist_ok=True)
os.makedirs(f"{SKILL_PATH}/references", exist_ok=True)
os.makedirs(f"{SKILL_PATH}/assets", exist_ok=True)

print(f"✅ 创建目录: {SKILL_PATH}/")
print(f"✅ 创建目录: {SKILL_PATH}/scripts/")
print(f"✅ 创建目录: {SKILL_PATH}/references/")
print(f"✅ 创建目录: {SKILL_PATH}/assets/")

# ========================================================================
# 创建 SKILL.md
# ========================================================================

skill_md = """---
name: medical-data-research
description: |
  医学数据全流程处理与研究论文生成。支持医疗记录数据（EMR/EHR）的清洗、分析、
  AI算法设计和医学核心期刊论文撰写。专门处理中文医疗术语、CSV格式数据、
  缺失数据补全、统计分析和论文材料生成。
  
  使用场景:
  1. 清洗和处理医疗记录CSV数据
  2. 设计医学AI算法流程（生存分析、风险预测、因果推断）
  3. 生成医学核心期刊论文材料（Abstract/Methods/Results/Discussion）
  4. 医学数据缺失模式分析和补全
  5. 与公开医学数据集（MIMIC、SEER等）对比验证
---

# Medical Data Research - 医学数据研究

全流程医学数据处理与研究论文生成工具。

## 使用场景

### 场景1: 医疗数据清洗与探索
用户: "帮我分析这个胰腺癌患者数据，计算缺失率"
→ 加载CSV → 缺失分析 → 生成统计图表

### 场景2: AI算法流程设计
用户: "设计一个预测胰腺癌患者生存期的模型"
→ 特征工程 → 模型选择 → 交叉验证 → 性能评估

### 场景3: 医学论文材料生成
用户: "基于这个数据分析结果写论文的Methods和Results"
→ 方法学描述 → 结果统计 → 图表说明 → 文献对比

### 场景4: 缺失数据补全
用户: "这个数据集缺失严重，用LLM补全一下"
→ 缺失模式分析 → 分层补全策略 → 验证 → 生成报告

## 核心功能

### 1. 数据加载与预处理
```python
# 使用 pandas 加载医疗记录CSV
df = pd.read_csv('patient_data.csv', encoding='utf-8')

# 处理中文医疗术语编码问题
df = handle_chinese_medical_terms(df)
```

**注意事项**:
- 检查CSV编码（通常是 UTF-8 或 GBK）
- 处理中文列名（建议转换为英文或拼音）
- 识别日期格式（常见：YYYY-MM-DD, YYYY/MM/DD）
- 处理多种缺失标记（空值、'NA'、'无'、'未记录'）

### 2. 医学数据特征分析

#### 2.1 缺失模式分析
使用 scripts/missing_analysis.py 进行完整的缺失分析：
- 特征级缺失率统计
- 患者级缺失分布
- 缺失模式热力图
- 模块级缺失对比（人口学/病史/检查/预后）

#### 2.2 数据质量评估
检查项目:
- 数值范围合理性（年龄 0-120，BMI 10-60）
- 时间线逻辑（诊断日期 ≤ 入院日期 ≤ 手术日期）
- 医学逻辑一致性（死亡患者应有生存时间）
- 与文献报道对比（性别比、合并症率）

### 3. AI算法设计流程

#### 3.1 生存分析（Survival Analysis）
适用场景: 预后预测、中位生存期估计

```python
from lifelines import CoxPHFitter, KaplanMeierFitter

# Cox比例风险模型
cph = CoxPHFitter()
cph.fit(df[features], duration_col='survival_days', event_col='is_deceased')

# Kaplan-Meier曲线
kmf = KaplanMeierFitter()
kmf.fit(df['survival_days'], event_observed=df['is_deceased'])
```

**评估指标**: C-index、Log-rank test、Brier score

#### 3.2 分类/预测模型
适用场景: 疾病诊断、并发症预测

推荐模型:
- XGBoost/LightGBM（表格数据首选）
- 逻辑回归（可解释性要求高）
- 深度学习（大规模影像-临床融合）

**医学AI特殊考虑**:
- 类别不平衡（阳性率通常<20%）
- 需要校准（Calibration）
- 可解释性（SHAP值、特征重要性）

#### 3.3 因果推断
适用场景: 治疗效果评估、风险因素分析

方法选择:
- 倾向评分匹配（PSM）
- 逆概率加权（IPW）
- 工具变量（IV）
- 双重差分（DiD，政策评估）

### 4. 医学论文材料生成

#### 4.1 论文结构模板
使用 assets/paper_template.md 作为起点：

```
标题: [疾病] [研究类型] [核心发现]

Abstract结构:
- Background: 疾病负担、研究缺口
- Objective: 研究目的
- Methods: 数据来源、样本量、分析方法
- Results: 主要发现（量化）
- Conclusions: 临床意义

Methods要点:
- 研究设计（回顾性/前瞻性）
- 纳入排除标准
- 统计方法（需具体到软件版本）
- 伦理审批
```

#### 4.2 期刊选择建议
根据研究类型选择目标期刊：
- **临床预测模型**: Journal of Clinical Oncology, Lancet Oncology
- **AI医学**: Nature Medicine, npj Digital Medicine
- **流行病学**: Cancer Epidemiology, International Journal of Cancer
- **专科**: Pancreas, HPB (专科期刊)

### 5. 缺失数据补全策略

#### 5.1 分层补全框架
```
Tier 1 - 规则补全（零成本）:
  - 文本解析（持续时间、日期）
  - 条件推断（is_smoker=0 → daily_smoking=0）
  - 衍生计算（BMI = weight/height²）

Tier 2 - LLM轻量补全（低成本）:
  - 症状关联推断
  - 临床分期推断
  - 家族史细节

Tier 3 - LLM深度推理（高成本）:
  - 合并症风险估计
  - 预后信息补全
```

#### 5.2 验证要求
- 内部一致性检查
- 与病理/影像结果对比
- 文献报道一致性
- 下游任务性能提升

## 脚本工具

### scripts/data_loader.py
标准化医疗数据加载器
- 自动检测编码
- 统一缺失值标记
- 数据类型推断

### scripts/missing_analysis.py
缺失分析完整流程
- 缺失率统计
- 可视化热力图
- 缺失模式聚类

### scripts/survival_analysis.py
生存分析完整流程
- Kaplan-Meier曲线
- Cox回归
- 竞争风险模型

### scripts/paper_generator.py
论文材料生成器
- Abstract生成
- Methods描述
- Results统计段落

## 参考文献

详细参考信息见:
- references/medical_datasets.md - 公开医学数据集汇总
- references/causal_methods.md - 因果推断方法对比
- references/statistical_guidelines.md - 医学统计指南

## 最佳实践

1. **数据隐私**: 确保患者ID已脱敏，符合HIPAA/网络安全法
2. **伦理审批**: 论文投稿前确认有IRB批件
3. **代码可重复**: 保存随机种子，记录软件版本
4. **结果验证**: 关键发现需与临床专家确认合理性
5. **图表规范**: 遵循目标期刊的图表格式要求
"""

# 写入 SKILL.md
with open(f"{SKILL_PATH}/SKILL.md", 'w', encoding='utf-8') as f:
    f.write(skill_md)

print(f"✅ 创建文件: {SKILL_PATH}/SKILL.md")

# ========================================================================
# 创建 scripts/data_loader.py
# ========================================================================

data_loader_script = '''#!/usr/bin/env python3
"""
医学数据标准化加载器
处理中文医疗术语、多种缺失标记、编码问题
"""

import pandas as pd
import numpy as np
import chardet

def detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']

def load_medical_data(file_path, **kwargs):
    """
    加载医学CSV数据，自动处理编码和缺失值
    
    Args:
        file_path: CSV文件路径
        **kwargs: 传递给pd.read_csv的其他参数
    
    Returns:
        DataFrame: 清洗后的数据
    """
    # 自动检测编码
    encoding = detect_encoding(file_path)
    print(f"检测到编码: {encoding}")
    
    # 加载数据
    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
    print(f"加载完成: {len(df)} 行 × {len(df.columns)} 列")
    
    # 处理中文列名（可选：转换为拼音）
    df = normalize_column_names(df)
    
    # 统一缺失值标记
    df = standardize_missing_values(df)
    
    return df

def normalize_column_names(df):
    """规范化列名"""
    # 保留原始列名映射
    original_cols = df.columns.tolist()
    
    # 清理列名（去除空格、特殊字符）
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    return df

def standardize_missing_values(df):
    """统一缺失值标记为标准NaN"""
    missing_markers = [
        '', 'NA', 'N/A', 'na', 'n/a',
        '无', '未知', '未记录', '未填写',
        'None', 'NULL', 'null',
        '0.0_units', '0_units'
    ]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(missing_markers, np.nan)
    
    return df

def parse_medical_duration(text_value):
    """解析医学持续时间文本"""
    import re
    
    if pd.isna(text_value):
        return np.nan
    
    text = str(text_value).strip()
    
    # 匹配 "360.0_years", "4.0_months" 等格式
    match = re.match(r'(\\d+\\.?\\d*)_(years?|months?|days?)', text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        
        converters = {
            'year': 12, 'years': 12,
            'month': 1, 'months': 1,
            'day': 1/30, 'days': 1/30
        }
        return value * converters.get(unit, 1)
    
    return np.nan

if __name__ == "__main__":
    # 示例用法
    df = load_medical_data("patient_data.csv")
    print(df.head())
'''

with open(f"{SKILL_PATH}/scripts/data_loader.py", 'w', encoding='utf-8') as f:
    f.write(data_loader_script)

print(f"✅ 创建文件: {SKILL_PATH}/scripts/data_loader.py")

# ========================================================================
# 创建 scripts/missing_analysis.py
# ========================================================================

missing_analysis_script = '''#!/usr/bin/env python3
"""
医学数据缺失分析完整流程
生成统计报告和可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def analyze_missing_comprehensive(df, output_dir="./output"):
    """
    全面的缺失分析
    
    Returns:
        dict: 包含统计结果和图表路径
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. 基本统计
    missing_matrix = df.isnull().astype(int)
    missing_rates = missing_matrix.mean().sort_values(ascending=False)
    
    results['total_cells'] = df.size
    results['missing_cells'] = missing_matrix.sum().sum()
    results['missing_rate_overall'] = results['missing_cells'] / results['total_cells']
    results['feature_missing_rates'] = missing_rates.to_dict()
    
    # 2. 患者级缺失
    patient_missing = missing_matrix.sum(axis=1)
    results['patient_missing_mean'] = patient_missing.mean()
    results['patient_missing_median'] = patient_missing.median()
    
    # 3. 生成热力图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 热力图1: 整体缺失
    ax1 = axes[0, 0]
    sample_patients = np.random.choice(df.index, min(100, len(df)), replace=False)
    key_features = [c for c in df.columns if missing_rates[c] < 0.9][:20]
    
    cmap = LinearSegmentedColormap.from_list('missing', ['white', '#d62728'])
    im = ax1.imshow(missing_matrix.loc[sample_patients, key_features].T, 
                    aspect='auto', cmap=cmap)
    ax1.set_title('Missing Value Heatmap (Sample Patients)')
    plt.colorbar(im, ax=ax1)
    
    # 保存
    plt.tight_layout()
    heatmap_path = f"{output_dir}/missing_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    results['heatmap_path'] = heatmap_path
    
    # 4. 保存详细报告
    report_path = f"{output_dir}/missing_report.csv"
    missing_df = pd.DataFrame({
        'feature': missing_rates.index,
        'missing_count': missing_matrix.sum(),
        'missing_rate': missing_rates.values
    })
    missing_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    results['report_path'] = report_path
    
    return results

if __name__ == "__main__":
    # 示例
    df = pd.read_csv("data.csv")
    results = analyze_missing_comprehensive(df)
    print(f"整体缺失率: {results['missing_rate_overall']:.2%}")
'''

with open(f"{SKILL_PATH}/scripts/missing_analysis.py", 'w', encoding='utf-8') as f:
    f.write(missing_analysis_script)

print(f"✅ 创建文件: {SKILL_PATH}/scripts/missing_analysis.py")

# ========================================================================
# 创建 references/medical_datasets.md
# ========================================================================

medical_datasets_ref = """# 公开医学数据集参考

## 重症医学

### MIMIC-IV
- **数据**: 38万+患者，6万+ICU住院
- **时间**: 2008-2022
- **内容**: 生命体征、实验室、药物、手术、临床记录
- **访问**: 需CITI培训、PhysioNet认证
- **用途**: 预后预测、治疗效果评估

### eICU-CRD
- **数据**: 20万+ICU住院，208家美国医院
- **内容**: 高频生理数据、APACHE评分
- **用途**: 多中心研究、基准测试

## 肿瘤

### SEER-Medicare
- **数据**: 450万+癌症患者
- **内容**: 癌症登记 + Medicare理赔
- **用途**: 比较效果研究、生存分析

### TCGA (The Cancer Genome Atlas)
- **数据**: 基因组 + 临床数据
- **用途**: 分子分型、基因-临床关联

## 通用健康

### NHANES
- **数据**: 每年5000人，连续调查
- **内容**: 营养、检查、实验室、问卷
- **访问**: 完全公开
- **用途**: 流行病学、风险因素

### UK Biobank
- **数据**: 50万+参与者
- **内容**: 基因、影像、健康记录
- **用途**: 孟德尔随机化、遗传风险

## 中国数据

### 中国健康与养老追踪调查 (CHARLS)
- **数据**: 1.7万+家庭
- **内容**: 健康、经济、家庭结构
- **用途**: 老龄化研究

### 中国肿瘤登记年报
- **来源**: 国家癌症中心
- **内容**: 发病率、死亡率、趋势
"""

with open(f"{SKILL_PATH}/references/medical_datasets.md", 'w', encoding='utf-8') as f:
    f.write(medical_datasets_ref)

print(f"✅ 创建文件: {SKILL_PATH}/references/medical_datasets.md")

# ========================================================================
# 创建 assets/paper_template.md
# ========================================================================

paper_template = """# 医学论文模板

## 标题格式
[研究对象] [研究设计] [核心方法] [主要发现]

示例:
- "Development and Validation of a Machine Learning Model for Predicting 
  Survival in Pancreatic Cancer Patients Using Electronic Health Records"
- "分层LLM增强框架用于胰腺癌电子病历缺失数据补全：基于下游预后分析的验证"

## Abstract 模板

### Background
[疾病]是一种[负担描述]，现有的[问题]。准确预测[结局]对[临床意义]。
然而，[数据挑战]。本研究旨在[目标]。

### Methods
这项回顾性研究纳入[样本量]例[患者类型]，来自[数据来源]。
我们开发了[方法]，包括[ tier 1/2/3 ]。
使用[统计/机器学习方法]进行[分析]。
通过[验证方法]评估性能。

### Results
队列平均年龄[XX]岁，[XX]%为男性。[关键发现1]。
[关键发现2，量化结果]。模型在[验证集]上表现[性能指标]。
与[基准/文献]相比[对比结果]。

### Conclusions
[方法]能有效[应用]。该工具可用于[临床意义]，
帮助[目标人群][行动]。

## Methods 模板

### 2.1 Study Design and Population
- 研究设计: 回顾性队列研究
- 时间范围: [开始日期] 至 [结束日期]
- 数据来源: [医院信息系统/EHR数据库]
- 纳入标准: [具体标准]
- 排除标准: [具体标准]
- 伦理审批: [IRB编号]

### 2.2 Variables
**暴露/特征变量**:
- 人口学: 年龄、性别、BMI
- 合并症: [ICD-10编码列表]
- 实验室: [指标列表]

**结局变量**:
- 主要结局: [定义]
- 次要结局: [定义]

### 2.3 Statistical Analysis
- 描述统计: 均值±标准差/中位数(IQR)
- 组间比较: t检验/Mann-Whitney U检验/卡方检验
- 生存分析: Kaplan-Meier, Cox回归
- 机器学习: [模型], [验证策略]
- 软件: Python 3.9, R 4.2, [包列表]

### 2.4 Missing Data Handling
- 缺失模式: [MCAR/MAR/MNAR假设]
- 处理方法: [插补方法/分层补全]
- 敏感性分析: [方法]

## Results 模板

### 3.1 Patient Characteristics
Table 1: [描述基线特征表]

### 3.2 Missing Data Patterns
[缺失分析结果，图表描述]

### 3.3 Model Performance
[性能指标，C-index, AUC等]

### 3.4 Feature Importance
[重要特征，SHAP值/HR]

### 3.5 Validation
[与文献对比，外部验证]

## Discussion 模板

### Principal Findings
[一句话总结主要发现]

### Comparison with Existing Literature
与[文献1]一致/差异...
与[文献2]相比...

### Clinical Implications
[对临床实践的意义]

### Strengths and Limitations
**优势**:
- 大样本量
- 多中心/长期随访
- 方法创新

**局限**:
- 回顾性设计
- 缺失数据
- 外部泛化性

### Future Directions
[未来研究方向]

## 投稿检查清单

- [ ] 格式符合目标期刊要求
- [ ] 图表分辨率≥300 DPI
- [ ] 参考文献格式正确
- [ ] 补充材料完整
- [ ] 数据可用性声明
- [ ] 利益冲突声明
- [ ] 伦理审批号
- [ ] 作者贡献声明
"""

with open(f"{SKILL_PATH}/assets/paper_template.md", 'w', encoding='utf-8') as f:
    f.write(paper_template)

print(f"✅ 创建文件: {SKILL_PATH}/assets/paper_template.md")

# ========================================================================
# 创建打包脚本
# ========================================================================

print("\n" + "="*80)
print("📦 打包 Skill")
print("="*80)

# 创建打包脚本
package_script = f'''#!/bin/bash
# 打包 {SKILL_NAME} skill

cd "{SKILL_PATH}/.."

# 创建临时目录
mkdir -p /tmp/{SKILL_NAME}
cp -r {SKILL_NAME}/* /tmp/{SKILL_NAME}/

# 打包为 .skill 文件 (zip格式)
cd /tmp
zip -r {SKILL_NAME}.skill {SKILL_NAME}/

# 移动到输出目录
mv {SKILL_NAME}.skill ~/Desktop/

echo "✅ 打包完成: ~/Desktop/{SKILL_NAME}.skill"
'''

with open(f"{SKILL_PATH}/../package_skill.sh", 'w') as f:
    f.write(package_script)

os.chmod(f"{SKILL_PATH}/../package_skill.sh", 0o755)

print(f"✅ 创建打包脚本: {SKILL_PATH}/../package_skill.sh")

# ========================================================================
# 总结
# ========================================================================

print("\n" + "="*80)
print(f"✨ Skill '{SKILL_NAME}' 创建完成!")
print("="*80)

print(f"""
📁 目录结构:
{SKILL_PATH}/
├── SKILL.md                      # 主文档 (已创建)
├── scripts/
│   ├── data_loader.py           # 数据加载器 (已创建)
│   └── missing_analysis.py      # 缺失分析 (已创建)
├── references/
│   └── medical_datasets.md      # 数据集参考 (已创建)
└── assets/
    └── paper_template.md        # 论文模板 (已创建)

📋 安装步骤:
1. 打包: bash {SKILL_PATH}/../package_skill.sh
2. 安装: openclaw skills install ~/Desktop/{SKILL_NAME}.skill
   或复制到: ~/.openclaw/skills/

🎯 使用示例:
用户: "帮我分析这个医疗数据"
→ 触发 medical-data-research skill
→ 自动加载数据、分析缺失、生成报告

用户: "写论文的Methods部分"
→ 使用 assets/paper_template.md
→ 生成符合医学期刊格式的Methods
""")
