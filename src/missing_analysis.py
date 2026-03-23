#!/usr/bin/env python3
"""
胰腺癌数据集缺失率统计分析
用于数据补全算法研究论文
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 读取数据
df = pd.read_csv('/Users/fanganita/.openclaw/media/inbound/pancreatic_cancer_data_normalized_clean---f3fc1dae-f432-4e85-a92d-62263d2ad843')

# 总样本数
total_patients = len(df)
print("="*80)
print("📊 胰腺癌数据集缺失率统计分析报告")
print("="*80)
print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"总样本量: {total_patients:,} 例患者")
print(f"总特征数: {len(df.columns)} 个")

# 定义特征分类
feature_categories = {
    '人口学信息': [
        'patient_id', 'gender', 'age', 'birth_date', 'marital_status', 
        'birthplace', 'blood_type', 'height_cm', 'weight_kg', 'bmi'
    ],
    '既往病史': [
        'has_hepatitis_b', 'hepatitis_b_duration', 'has_alcoholic_liver', 
        'alcoholic_liver_duration', 'has_biliary_inflammation', 'has_gallstones',
        'has_gallbladder_polyps', 'has_acute_pancreatitis', 'has_chronic_pancreatitis',
        'has_diabetes', 'diabetes_duration', 'has_hypertension', 'has_hyperlipidemia',
        'has_coronary_disease', 'has_hp_infection', 'has_eb_virus'
    ],
    '生活习惯': [
        'is_smoker', 'daily_smoking_amount', 'smoking_duration', 'has_quit_smoking',
        'quit_smoking_date', 'is_drinker', 'daily_drinking_amount', 'drinking_duration',
        'has_quit_drinking', 'quit_drinking_date'
    ],
    '家族史': [
        'family_pancreatic_cancer', 'family_pancreatic_cancer_relation',
        'family_other_cancer', 'family_obesity'
    ],
    '临床表现': [
        'has_abdominal_pain', 'has_abdominal_distension', 'has_jaundice',
        'jaundice_treatment', 'has_nausea', 'has_vomiting', 'has_back_pain',
        'has_diarrhea', 'has_weight_loss', 'has_hypoglycemia', 'has_peptic_ulcer',
        'ecog_score', 'nrs_score', 'pancreatic_cancer_duration'
    ],
    '实验室检查': [
        'fasting_glucose', 'hba1c', 'fasting_glucose_mmol_L', 'hba1c_percent',
        'tb_before_treatment', 'db_before_treatment', 'ggt_before_treatment',
        'tb_after_treatment', 'db_after_treatment', 'ggt_after_treatment'
    ],
    '诊断与分期': [
        'diagnosis_date', 'pathologically_confirmed', 'has_mdt_discussion',
        'mdt_during_treatment', 'stage_at_diagnosis', 'clinical_stage',
        'staging_system', 'clinical_t_stage', 'clinical_n_stage', 'clinical_m_stage',
        'pathological_stage', 'pathological_t_stage', 'pathological_n_stage',
        'pathological_m_stage'
    ],
    '治疗信息': [
        'specimen_acquisition_method', 'surgery_method', 'surgery_name', 'surgery_date'
    ],
    '预后随访': [
        'is_deceased', 'death_date', 'cause_of_death', 'first_followup_date',
        'last_followup_date', 'has_recurrence', 'has_metastasis', 'metastasis_site',
        'liver_metastases_count', 'lung_metastases_count', 'survival_days', 'survival_months'
    ],
    '转移部位': [
        'metastasis_to_Liver', 'metastasis_to_Lung', 'metastasis_to_Bone',
        'metastasis_to_Peritoneum', 'metastasis_to_Lymph_node', 
        'metastasis_to_Adrenal', 'metastasis_to_Brain'
    ],
    '衍生变量': [
        'pancreatic_cancer_duration_months', 'pancreatic_cancer_duration_text',
        'hepatitis_b_duration_months', 'hepatitis_b_duration_text',
        'alcoholic_liver_duration_months', 'alcoholic_liver_duration_text',
        'diabetes_duration_months', 'diabetes_duration_text',
        'smoking_duration_months', 'smoking_duration_text',
        'drinking_duration_months', 'drinking_duration_text',
        'daily_smoking_cigarettes', 'daily_alcohol_g',
        'weight_change_description', 'has_weight_change',
        'data_completeness_score'
    ]
}

# 统计每个特征的缺失情况
def analyze_missing(df, feature_list, category_name):
    """分析指定类别特征的缺失情况"""
    results = []
    available_features = [f for f in feature_list if f in df.columns]
    
    for feat in available_features:
        series = df[feat]
        
        # 计算各种形式的缺失
        null_count = series.isnull().sum()
        
        # 检查空字符串、'0.0_units'等特殊缺失标记
        empty_str = (series == '').sum() if series.dtype == 'object' else 0
        na_str = (series == 'NA').sum() if series.dtype == 'object' else 0
        unknown_str = (series.astype(str).str.lower() == 'unknown').sum()
        
        # 特殊缺失标记（根据数据观察）
        special_missing = 0
        if series.dtype == 'object':
            special_missing += (series == '0.0_units').sum()
            special_missing += (series == '0.0').sum()
            special_missing += (series == '0').sum()
            special_missing += (series == 'None').sum()
            special_missing += series.astype(str).str.contains('units', na=False).sum()
        
        total_missing = null_count + empty_str + na_str + unknown_str + special_missing
        missing_rate = total_missing / len(df) * 100
        
        results.append({
            '特征名': feat,
            '类别': category_name,
            '缺失数': total_missing,
            '缺失率(%)': round(missing_rate, 2),
            '数据类型': str(series.dtype),
            '唯一值数': series.nunique(),
            '示例值': str(series.dropna().iloc[0]) if not series.dropna().empty else 'N/A'
        })
    
    return results

# 执行分析
all_results = []
category_stats = {}

print("\n" + "="*80)
print("📈 各模块缺失率统计")
print("="*80)

for category, features in feature_categories.items():
    results = analyze_missing(df, features, category)
    if results:
        all_results.extend(results)
        
        # 计算该类别统计
        missing_rates = [r['缺失率(%)'] for r in results]
        category_stats[category] = {
            '特征数': len(results),
            '平均缺失率': np.mean(missing_rates),
            '最高缺失率': max(missing_rates),
            '最低缺失率': min(missing_rates),
            '完全缺失(>95%)': sum(1 for r in results if r['缺失率(%)'] > 95),
            '严重缺失(>50%)': sum(1 for r in results if r['缺失率(%)'] > 50),
            '中度缺失(20-50%)': sum(1 for r in results if 20 <= r['缺失率(%)'] <= 50),
            '轻度缺失(<20%)': sum(1 for r in results if r['缺失率(%)'] < 20)
        }
        
        print(f"\n【{category}】")
        print(f"  特征数量: {len(results)}")
        print(f"  平均缺失率: {np.mean(missing_rates):.2f}%")
        print(f"  最高缺失率: {max(missing_rates):.2f}% ({max(results, key=lambda x: x['缺失率(%)'])['特征名']})")
        print(f"  缺失分级: 严重(>50%):{category_stats[category]['严重缺失(>50%)']}, "
              f"中度(20-50%):{category_stats[category]['中度缺失(20-50%)']}, "
              f"轻度(<20%):{category_stats[category]['轻度缺失(<20%)']}")

# 创建详细结果DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('缺失率(%)', ascending=False)

print("\n" + "="*80)
print("🔴 缺失率最高的20个特征")
print("="*80)
print(results_df.head(20)[['特征名', '类别', '缺失率(%)', '数据类型']].to_string(index=False))

print("\n" + "="*80)
print("🟢 最完整的20个特征（缺失率最低）")
print("="*80)
print(results_df.tail(20)[['特征名', '类别', '缺失率(%)', '数据类型']].to_string(index=False))

# 整体统计
print("\n" + "="*80)
print("📊 整体数据质量评估")
print("="*80)

all_missing_rates = results_df['缺失率(%)'].values
print(f"总体特征平均缺失率: {np.mean(all_missing_rates):.2f}%")
print(f"总体特征中位缺失率: {np.median(all_missing_rates):.2f}%")
print(f"标准差: {np.std(all_missing_rates):.2f}%")

# 缺失率分级统计
severe = sum(all_missing_rates > 80)
high = sum((all_missing_rates > 50) & (all_missing_rates <= 80))
medium = sum((all_missing_rates > 20) & (all_missing_rates <= 50))
low = sum(all_missing_rates <= 20)

print(f"\n缺失率分布:")
print(f"  🔴 严重缺失 (>80%): {severe} 个特征 ({severe/len(all_missing_rates)*100:.1f}%)")
print(f"  🟠 高度缺失 (50-80%): {high} 个特征 ({high/len(all_missing_rates)*100:.1f}%)")
print(f"  🟡 中度缺失 (20-50%): {medium} 个特征 ({medium/len(all_missing_rates)*100:.1f}%)")
print(f"  🟢 轻度缺失 (<20%): {low} 个特征 ({low/len(all_missing_rates)*100:.1f}%)")

# 计算每个患者的缺失特征数
df_numeric = df.copy()
# 将各种缺失标记转换为NaN
for col in df_numeric.columns:
    if df_numeric[col].dtype == 'object':
        df_numeric[col] = df_numeric[col].replace({
            '': np.nan, 'NA': np.nan, 'None': np.nan, 'unknown': np.nan,
            '0.0_units': np.nan, '0.0': np.nan
        })
        # 包含units的视为缺失
        mask = df_numeric[col].astype(str).str.contains('units', na=False)
        df_numeric.loc[mask, col] = np.nan

patient_missing = df_numeric.isnull().sum(axis=1)
print(f"\n患者维度缺失分析:")
print(f"  平均每例患者缺失特征数: {patient_missing.mean():.1f} / {len(df.columns)}")
print(f"  中位数: {patient_missing.median():.1f}")
print(f"  最少缺失: {patient_missing.min()} 个特征")
print(f"  最多缺失: {patient_missing.max()} 个特征")

# 完全缺失的样本
complete_missing = sum(patient_missing == len(df.columns))
print(f"  完全无数据的样本: {complete_missing} 例")

# 保存详细结果
results_df.to_csv('/Users/fanganita/.openclaw/workspace/缺失率统计_详细结果.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 详细结果已保存至: 缺失率统计_详细结果.csv")

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 图1: 各模块平均缺失率对比
cat_names = list(category_stats.keys())
cat_means = [category_stats[cat]['平均缺失率'] for cat in cat_names]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cat_names)))

ax1 = axes[0, 0]
bars = ax1.barh(cat_names, cat_means, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Missing Rate (%)', fontsize=11)
ax1.set_title('Average Missing Rate by Category\n(各模块平均缺失率)', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 100)
for bar, val in zip(bars, cat_means):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
             va='center', fontsize=9)
ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax1.grid(axis='x', alpha=0.3)

# 图2: 缺失率分布直方图
ax2 = axes[0, 1]
ax2.hist(all_missing_rates, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=np.mean(all_missing_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_missing_rates):.1f}%')
ax2.axvline(x=np.median(all_missing_rates), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(all_missing_rates):.1f}%')
ax2.set_xlabel('Missing Rate (%)', fontsize=11)
ax2.set_ylabel('Number of Features', fontsize=11)
ax2.set_title('Distribution of Missing Rates\n(缺失率分布)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 图3: Top 20 最高缺失率特征
ax3 = axes[1, 0]
top20 = results_df.head(20)
colors_top20 = ['#d62728' if x > 80 else '#ff7f0e' if x > 50 else '#2ca02c' for x in top20['缺失率(%)']]
ax3.barh(range(20), top20['缺失率(%)'].values, color=colors_top20, edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(20))
ax3.set_yticklabels([f"{name[:25]}..." if len(name) > 25 else name 
                      for name in top20['特征名'].values], fontsize=8)
ax3.invert_yaxis()
ax3.set_xlabel('Missing Rate (%)', fontsize=11)
ax3.set_title('Top 20 Features with Highest Missing Rate\n(缺失率最高的20个特征)', fontsize=13, fontweight='bold')
ax3.set_xlim(0, 100)
ax3.axvline(x=50, color='red', linestyle='--', alpha=0.5)
ax3.grid(axis='x', alpha=0.3)

# 图4: 缺失率分级饼图
ax4 = axes[1, 1]
labels_pie = ['Severe\n(>80%)', 'High\n(50-80%)', 'Medium\n(20-50%)', 'Low\n(<20%)']
sizes_pie = [severe, high, medium, low]
colors_pie = ['#d62728', '#ff7f0e', '#ffdd44', '#2ca02c']
explode = (0.05, 0.02, 0.02, 0.02)

wedges, texts, autotexts = ax4.pie(sizes_pie, explode=explode, labels=labels_pie, 
                                     colors=colors_pie, autopct='%1.1f%%',
                                     shadow=True, startangle=90)
ax4.set_title('Missing Rate Classification\n(缺失率分级分布)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/fanganita/.openclaw/workspace/缺失率统计_可视化图.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/fanganita/.openclaw/workspace/缺失率统计_可视化图.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 可视化图表已保存至: 缺失率统计_可视化图.png/pdf")

# 生成论文可用的统计描述
print("\n" + "="*80)
print("📝 论文可用的数据描述段落")
print("="*80)

paper_text = f"""
【数据质量描述 - 可直接用于论文】

本研究纳入胰腺癌患者共 {total_patients} 例，数据集包含 {len(df.columns)} 个临床特征变量，
涵盖人口学信息、既往病史、临床表现、实验室检查、诊断分期、治疗方式及预后随访等多个维度。

数据质量评估显示，特征变量的平均缺失率为 {np.mean(all_missing_rates):.1f}% 
(标准差 {np.std(all_missing_rates):.1f}%)，中位缺失率为 {np.median(all_missing_rates):.1f}%。
按缺失程度分级：严重缺失(>80%) {severe} 个({severe/len(all_missing_rates)*100:.1f}%)、
高度缺失(50-80%) {high} 个({high/len(all_missing_rates)*100:.1f}%)、
中度缺失(20-50%) {medium} 个({medium/len(all_missing_rates)*100:.1f}%)、
轻度缺失(<20%) {low} 个({low/len(all_missing_rates)*100:.1f}%)。

从特征类别看，平均缺失率最高的模块为：
"""

# 找出缺失率最高的3个类别
sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]['平均缺失率'], reverse=True)
for i, (cat, stats) in enumerate(sorted_cats[:3], 1):
    paper_text += f"{i}. {cat} ({stats['平均缺失率']:.1f}%)\n"

paper_text += f"""
患者维度的缺失分析显示，平均每例患者缺失 {patient_missing.mean():.1f} 个特征变量，
中位数为 {patient_missing.median():.1f} 个。这一高缺失率特征充分反映了真实医疗数据
"脏乱差"的现状，为验证本研究提出的LLM驱动数据补全算法提供了典型的应用场景。
"""

print(paper_text)

# 保存论文文本
with open('/Users/fanganita/.openclaw/workspace/论文可用_数据描述文本.txt', 'w', encoding='utf-8') as f:
    f.write(paper_text)
print("\n✅ 论文描述文本已保存至: 论文可用_数据描述文本.txt")

print("\n" + "="*80)
print("✨ 分析完成！")
print("="*80)
print("\n生成的文件:")
print("  1. 缺失率统计_详细结果.csv - 每个特征的详细缺失统计")
print("  2. 缺失率统计_可视化图.png/pdf - 用于论文的图表")
print("  3. 论文可用_数据描述文本.txt - 可直接使用的论文段落")
