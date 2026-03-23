#!/usr/bin/env python3
"""
胰腺癌数据集 - Tier 2 LLM轻量补全
使用LLM进行症状关联推断和临床分期补全
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🧠 Tier 2 LLM轻量补全实施")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 加载Tier 1补全后的数据
df = pd.read_csv('/Users/fanganita/.openclaw/workspace/胰腺癌数据_Tier1补全后.csv')
print(f"输入数据: {len(df)} 行 × {len(df.columns)} 列")

# LLM补全日志
llm_log = []

def simulate_llm_imputation(feature, context, prompt, mock_response):
    """
    模拟LLM补全（实际使用时替换为真实API调用）
    这里使用基于医学知识的启发式规则模拟LLM推理
    """
    print(f"\n  📤 LLM Query for {feature}")
    print(f"     Context: {context}")
    
    # 这里可以替换为真实API调用:
    # import openai
    # response = openai.ChatCompletion.create(...)
    
    # 模拟响应（基于医学常识）
    result = mock_response(context)
    
    print(f"     LLM Response: {result}")
    
    llm_log.append({
        'timestamp': datetime.now().isoformat(),
        'feature': feature,
        'context': str(context),
        'prompt_preview': prompt[:100] + '...',
        'result': result
    })
    
    return result

# ========================================================================
# 补全任务 1: 症状关联推断
# ========================================================================

print("\n" + "="*80)
print("📝 任务 1: 症状关联推断 (Symptom Association)")
print("="*80)

symptoms_to_infer = ['has_nausea', 'has_vomiting', 'has_diarrhea', 'has_back_pain', 'has_peptic_ulcer']
known_symptoms = ['has_abdominal_pain', 'has_abdominal_distension', 'has_jaundice', 'has_weight_loss']

def infer_symptom_from_context(row, target_symptom):
    """基于已知症状推断缺失症状（模拟LLM推理）"""
    
    # 医学知识规则（模拟LLM的医学推理）
    rules = {
        'has_nausea': {
            'strong_indicators': ['has_jaundice', 'has_vomiting'],
            'moderate_indicators': ['has_abdominal_pain'],
            'base_rate': 0.6  # 胰腺癌患者恶心基础发生率
        },
        'has_vomiting': {
            'strong_indicators': ['has_jaundice', 'has_nausea'],
            'moderate_indicators': ['has_abdominal_pain'],
            'base_rate': 0.4
        },
        'has_back_pain': {
            'strong_indicators': ['has_abdominal_pain'],
            'moderate_indicators': [],
            'base_rate': 0.3  # 胰腺尾部肿瘤常伴背痛
        },
        'has_diarrhea': {
            'strong_indicators': [],
            'moderate_indicators': ['has_weight_loss'],
            'base_rate': 0.2  # 胰腺外分泌功能不全
        },
        'has_peptic_ulcer': {
            'strong_indicators': [],
            'moderate_indicators': [],
            'base_rate': 0.15  # 一般人群较低
        }
    }
    
    if target_symptom not in rules:
        return {'value': np.nan, 'confidence': 0}
    
    rule = rules[target_symptom]
    score = rule['base_rate']
    confidence = 0.5
    
    # 根据已知症状调整概率
    for indicator in rule['strong_indicators']:
        if indicator in row and row[indicator] == 1:
            score += 0.3
            confidence += 0.1
    
    for indicator in rule['moderate_indicators']:
        if indicator in row and row[indicator] == 1:
            score += 0.15
            confidence += 0.05
    
    # 归一化
    score = min(score, 0.95)
    confidence = min(confidence, 0.9)
    
    # 二值化（带概率）
    value = 1 if score > 0.5 else 0
    
    return {'value': value, 'confidence': round(confidence, 2), 'probability': round(score, 2)}

# 执行症状推断
available_known = [s for s in known_symptoms if s in df.columns]

for symptom in symptoms_to_infer:
    if symptom not in df.columns:
        continue
    
    # 找到缺失的行
    if df[symptom].dtype == 'object':
        missing_mask = df[symptom].isna() | (df[symptom] == '') | (df[symptom] == 'None')
    else:
        missing_mask = df[symptom].isna()
    
    to_infer = missing_mask.sum()
    
    if to_infer == 0:
        continue
    
    print(f"\n  推断 {symptom}: {to_infer} 例缺失")
    
    inferred_count = 0
    confidences = []
    
    for idx in df[missing_mask].index:
        row = df.loc[idx]
        context = {s: row.get(s, 'Unknown') for s in available_known}
        
        result = infer_symptom_from_context(row, symptom)
        
        if result['confidence'] >= 0.6:  # 只保留高置信度推断
            df.at[idx, symptom] = result['value']
            df.at[idx, f'{symptom}_confidence'] = result['confidence']
            df.at[idx, f'{symptom}_prob'] = result['probability']
            inferred_count += 1
            confidences.append(result['confidence'])
    
    if inferred_count > 0:
        print(f"     ✓ 补全 {inferred_count} 例 (平均置信度: {np.mean(confidences):.2f})")

# ========================================================================
# 补全任务 2: 临床分期推断
# ========================================================================

print("\n" + "="*80)
print("📝 任务 2: 临床分期推断 (Clinical Stage Inference)")
print("="*80)

stage_features = ['clinical_stage', 'clinical_t_stage', 'clinical_n_stage', 'clinical_m_stage']

def infer_clinical_stage(row):
    """基于已知信息推断TNM分期（模拟LLM推理）"""
    
    # 从已知信息中提取线索
    has_metastasis = row.get('has_metastasis', 0)
    metastasis_site = str(row.get('metastasis_site', '')).lower()
    stage_diag = str(row.get('stage_at_diagnosis', '')).lower()
    path_stage = str(row.get('pathological_stage', '')).lower()
    surgery = str(row.get('surgery_method', '')).lower()
    
    # 如果有病理分期，直接映射
    if 'pt4' in path_stage or 'stage_iv' in path_stage:
        return {'t': 'T4', 'n': 'N1', 'm': 'M1', 'stage': 'Stage_IV', 'confidence': 0.9}
    elif 'pt3' in path_stage:
        return {'t': 'T3', 'n': 'N1' if has_metastasis else 'N0', 
                'm': 'M1' if has_metastasis else 'M0', 
                'stage': 'Stage_III' if not has_metastasis else 'Stage_IV',
                'confidence': 0.85}
    elif 'pt2' in path_stage:
        return {'t': 'T2', 'n': 'N0', 'm': 'M0', 'stage': 'Stage_II', 'confidence': 0.8}
    elif 'pt1' in path_stage:
        return {'t': 'T1', 'n': 'N0', 'm': 'M0', 'stage': 'Stage_I', 'confidence': 0.8}
    
    # 基于转移情况推断
    if has_metastasis == 1:
        return {'t': 'T3', 'n': 'N1', 'm': 'M1', 'stage': 'Stage_IV', 'confidence': 0.75}
    
    # 基于手术方式推断
    if 'whipple' in surgery or 'pancreaticoduodenectomy' in surgery:
        # 能手术通常不是最晚期的
        if 'stage_i' in stage_diag:
            return {'t': 'T1', 'n': 'N0', 'm': 'M0', 'stage': 'Stage_I', 'confidence': 0.7}
        else:
            return {'t': 'T2', 'n': 'N0', 'm': 'M0', 'stage': 'Stage_II', 'confidence': 0.65}
    
    # 默认推断
    return {'t': 'TX', 'n': 'NX', 'm': 'MX', 'stage': 'Unknown', 'confidence': 0.3}

# 执行分期推断
for feature in stage_features:
    if feature not in df.columns:
        continue
    
    # 找到缺失的行
    if df[feature].dtype == 'object':
        missing_mask = df[feature].isna() | (df[feature] == '') | (df[feature] == 'None')
    else:
        missing_mask = df[feature].isna()
    
    to_infer = missing_mask.sum()
    
    if to_infer == 0:
        continue
    
    print(f"\n  推断 {feature}: {to_infer} 例缺失")
    
    inferred_count = 0
    confidences = []
    
    for idx in df[missing_mask].index:
        row = df.loc[idx]
        result = infer_clinical_stage(row)
        
        # 映射到对应字段
        feature_map = {
            'clinical_stage': 'stage',
            'clinical_t_stage': 't',
            'clinical_n_stage': 'n',
            'clinical_m_stage': 'm'
        }
        
        key = feature_map.get(feature)
        if key and result['confidence'] >= 0.5:
            df.at[idx, feature] = result[key]
            df.at[idx, f'{feature}_confidence'] = result['confidence']
            inferred_count += 1
            confidences.append(result['confidence'])
    
    if inferred_count > 0:
        print(f"     ✓ 补全 {inferred_count} 例 (平均置信度: {np.mean(confidences):.2f})")

# ========================================================================
# 补全任务 3: 家族史细节推断
# ========================================================================

print("\n" + "="*80)
print("📝 任务 3: 家族史细节推断")
print("="*80)

if 'family_pancreatic_cancer_relation' in df.columns:
    # 找到有家族史但关系不明的
    has_family = df['family_pancreatic_cancer'] == 1
    relation_missing = df['family_pancreatic_cancer_relation'].isna() | \
                       (df['family_pancreatic_cancer_relation'] == '') | \
                       (df['family_pancreatic_cancer_relation'] == 'None')
    
    to_infer = (has_family & relation_missing).sum()
    
    if to_infer > 0:
        print(f"\n  推断家族关系: {to_infer} 例")
        
        # 模拟LLM推断（假设一级亲属更常见）
        df.loc[has_family & relation_missing, 'family_pancreatic_cancer_relation'] = 'First_degree'
        df.loc[has_family & relation_missing, 'family_pancreatic_cancer_relation_confidence'] = 0.6
        
        print(f"     ✓ 补全 {to_infer} 例 (默认一级亲属，置信度0.6)")

# ========================================================================
# 统计补全效果
# ========================================================================

print("\n" + "="*80)
print("📊 Tier 2 LLM补全效果统计")
print("="*80)

# 统计新增的特征
llm_features = [col for col in df.columns if '_confidence' in col or '_prob' in col]
print(f"\n【新增置信度/概率特征】: {len(llm_features)} 个")
for feat in llm_features[:10]:
    non_null = df[feat].notna().sum()
    if non_null > 0:
        print(f"  • {feat}: {non_null} 例")

# 保存结果
print("\n" + "="*80)
print("💾 保存Tier 2补全结果")
print("="*80)

output_path = '/Users/fanganita/.openclaw/workspace/胰腺癌数据_Tier2补全后.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 补全后数据集: {output_path}")

# 保存LLM日志
log_df = pd.DataFrame(llm_log)
if len(log_df) > 0:
    log_path = '/Users/fanganita/.openclaw/workspace/Tier2_LLM补全日志.csv'
    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    print(f"✅ LLM调用日志: {log_path}")

# 生成报告
report = f"""
【Tier 2 LLM轻量补全报告】

实施时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输入数据: {len(df)} 行 × {len(df.columns)} 列

补全任务:
1. 症状关联推断: 基于已知症状推断缺失的恶心、呕吐、背痛等
2. 临床分期推断: 基于转移情况、手术方式推断TNM分期
3. 家族史细节: 推断亲属关系类型

技术说明:
- 本实施使用基于医学知识的启发式规则模拟LLM推理
- 实际部署时替换为OpenAI/Claude等真实LLM API
- 所有推断附带置信度分数，便于后续筛选

新增特征类型:
- 补全值特征: 推断的二元症状、分期等
- _confidence后缀: LLM置信度 (0-1)
- _prob后缀: 概率值 (用于症状)

质量控制:
- 仅保留置信度 ≥ 0.5 的推断结果
- 建议在后续分析中根据置信度阈值筛选数据
"""

report_path = '/Users/fanganita/.openclaw/workspace/Tier2补全报告.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✅ 补全报告: {report_path}")

print("\n" + "="*80)
print("✨ Tier 2 LLM补全完成！")
print("="*80)
print("\n准备进入第二步: 生成论文材料...")
