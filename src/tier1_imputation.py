#!/usr/bin/env python3
"""
胰腺癌数据集 - Tier 1 规则补全实施
零成本、高确定性的数据补全
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("🔧 Tier 1 规则补全实施")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 加载数据
df = pd.read_csv('/Users/fanganita/.openclaw/media/inbound/pancreatic_cancer_data_normalized_clean---f3fc1dae-f432-4e85-a92d-62263d2ad843')
print(f"原始数据: {len(df)} 行 × {len(df.columns)} 列")

# 记录补全日志
imputation_log = []

def log_imputation(feature, method, condition, count, example=None):
    """记录补全操作"""
    imputation_log.append({
        'timestamp': datetime.now().isoformat(),
        'feature': feature,
        'method': method,
        'condition': condition,
        'count': count,
        'example': example
    })
    print(f"  ✓ {feature}: {method} | {condition} | 补全{count}条")

# 创建缺失统计函数
def calc_missing_rate(series):
    """计算缺失率，处理多种缺失形式"""
    if series.dtype == 'object':
        missing = series.isna() | (series == '') | (series == 'None') | \
                  (series == '0.0_units') | (series == '0.0') | (series == '0') | \
                  series.astype(str).str.contains('units', na=False)
    else:
        missing = series.isna()
    return missing.sum(), missing.mean() * 100

# 初始化补全前统计
print("\n" + "="*80)
print("📊 补全前缺失统计")
print("="*80)

initial_stats = {}
for col in df.columns:
    missing_count, missing_rate = calc_missing_rate(df[col])
    initial_stats[col] = {'count': missing_count, 'rate': missing_rate}

total_cells = len(df) * len(df.columns)
total_missing = sum(s['count'] for s in initial_stats.values())
print(f"总缺失单元格: {total_missing:,} / {total_cells:,} ({total_missing/total_cells*100:.2f}%)")

# ========================================================================
# 补全任务 1: 持续时间文本解析
# ========================================================================

print("\n" + "="*80)
print("📝 任务 1: 持续时间文本解析")
print("="*80)

def parse_duration_to_months(val):
    """解析持续时间文本为月数"""
    if pd.isna(val) or str(val) in ['', 'None', 'nan', '0.0_units', '0.0', '0']:
        return np.nan
    
    val_str = str(val).strip()
    
    # 匹配数字+单位的模式
    match = re.match(r'(\d+\.?\d*)_(years|months|days|year|month|day)s?', val_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        
        converters = {
            'year': 12,
            'years': 12,
            'month': 1,
            'months': 1,
            'day': 1/30,
            'days': 1/30
        }
        return value * converters.get(unit, 1)
    
    return np.nan

# 处理持续时间字段
duration_fields = [
    ('hepatitis_b_duration', 'hepatitis_b_duration_months'),
    ('alcoholic_liver_duration', 'alcoholic_liver_duration_months'),
    ('diabetes_duration', 'diabetes_duration_months'),
    ('smoking_duration', 'smoking_duration_months'),
    ('drinking_duration', 'drinking_duration_months'),
    ('pancreatic_cancer_duration', 'pancreatic_cancer_duration_months')
]

for source_col, target_col in duration_fields:
    if source_col in df.columns:
        # 计算补全前缺失
        before_missing, before_rate = calc_missing_rate(df[source_col])
        
        # 执行解析
        df[target_col] = df[source_col].apply(parse_duration_to_months)
        
        # 计算有效补全数
        after_valid = df[target_col].notna().sum()
        filled_count = after_valid - (len(df) - before_missing)
        
        if filled_count > 0:
            example = df[df[target_col].notna()][source_col].iloc[0] if after_valid > 0 else None
            log_imputation(target_col, 'Duration Parsing', f'{source_col} → months', 
                          filled_count, example)

# ========================================================================
# 补全任务 2: 条件零值推断
# ========================================================================

print("\n" + "="*80)
print("📝 任务 2: 条件零值推断")
print("="*80)

# 定义条件规则
conditional_rules = [
    # (条件列, 条件值, 目标列, 填充值, 描述)
    ('is_smoker', 0, 'daily_smoking_amount', 0, '非吸烟者每日吸烟量=0'),
    ('is_smoker', 0, 'daily_smoking_cigarettes', 0, '非吸烟者每日吸烟支数=0'),
    ('is_drinker', 0, 'daily_drinking_amount', 0, '非饮酒者每日饮酒量=0'),
    ('is_drinker', 0, 'daily_alcohol_g', 0, '非饮酒者每日酒精克数=0'),
]

# 处理日期字段的特殊规则
date_rules = [
    ('has_quit_smoking', 0, 'quit_smoking_date', pd.NaT, '未戒烟者戒烟日期=None'),
    ('has_quit_drinking', 0, 'quit_drinking_date', pd.NaT, '未戒酒者戒酒日期=None'),
]

# 执行数值规则
for condition_col, condition_val, target_col, fill_val, desc in conditional_rules:
    if condition_col in df.columns and target_col in df.columns:
        # 找到满足条件且目标缺失的行
        condition_mask = (df[condition_col] == condition_val)
        
        # 对于目标列，需要判断是否为缺失
        if df[target_col].dtype == 'object':
            target_missing = df[target_col].isna() | (df[target_col] == '') | \
                           (df[target_col] == '0.0_units') | (df[target_col] == '0.0') | \
                           (df[target_col] == '0') | (df[target_col] == 'None')
        else:
            target_missing = df[target_col].isna()
        
        to_fill = condition_mask & target_missing
        filled_count = to_fill.sum()
        
        if filled_count > 0:
            df.loc[to_fill, target_col] = fill_val
            log_imputation(target_col, 'Conditional Zero', 
                          f'{condition_col}={condition_val}', filled_count)

# 执行日期规则
for condition_col, condition_val, target_col, fill_val, desc in date_rules:
    if condition_col in df.columns and target_col in df.columns:
        condition_mask = (df[condition_col] == condition_val)
        target_missing = df[target_col].isna() | (df[target_col] == '')
        to_fill = condition_mask & target_missing
        filled_count = to_fill.sum()
        
        if filled_count > 0:
            df.loc[to_fill, target_col] = fill_val
            log_imputation(target_col, 'Conditional Null', 
                          f'{condition_col}={condition_val}', filled_count)

# ========================================================================
# 补全任务 3: 衍生计算
# ========================================================================

print("\n" + "="*80)
print("📝 任务 3: 衍生计算")
print("="*80)

# BMI计算
if all(c in df.columns for c in ['weight_kg', 'height_cm', 'bmi']):
    # 找到bmi缺失但有身高体重的行
    weight_valid = pd.to_numeric(df['weight_kg'], errors='coerce').notna()
    height_valid = pd.to_numeric(df['height_cm'], errors='coerce').notna()
    
    if df['bmi'].dtype == 'object':
        bmi_missing = df['bmi'].isna() | (df['bmi'] == '') | (df['bmi'] == 'None')
    else:
        bmi_missing = df['bmi'].isna()
    
    to_calc = bmi_missing & weight_valid & height_valid
    filled_count = to_calc.sum()
    
    if filled_count > 0:
        weight = pd.to_numeric(df.loc[to_calc, 'weight_kg'], errors='coerce')
        height = pd.to_numeric(df.loc[to_calc, 'height_cm'], errors='coerce')
        df.loc[to_calc, 'bmi'] = weight / ((height / 100) ** 2)
        log_imputation('bmi', 'Formula: w/(h/100)^2', 
                      'weight_kg + height_cm available', filled_count)

# 病程月数计算 (从日期)
if all(c in df.columns for c in ['admission_date', 'diagnosis_date']):
    # 转换日期
    df['admission_date_dt'] = pd.to_datetime(df['admission_date'], errors='coerce')
    df['diagnosis_date_dt'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
    
    # 找到可以计算的
    both_dates_valid = df['admission_date_dt'].notna() & df['diagnosis_date_dt'].notna()
    
    target_col = 'pancreatic_cancer_duration_months_calc'
    if target_col not in df.columns:
        df[target_col] = np.nan
    
    target_missing = df[target_col].isna()
    to_calc = both_dates_valid & target_missing
    filled_count = to_calc.sum()
    
    if filled_count > 0:
        duration_days = (df.loc[to_calc, 'admission_date_dt'] - 
                        df.loc[to_calc, 'diagnosis_date_dt']).dt.days
        df.loc[to_calc, target_col] = duration_days / 30.44  # 平均每月天数
        log_imputation(target_col, 'Date Diff: (admission - diagnosis)/30.44', 
                      'Both dates available', filled_count)

# ========================================================================
# 补全任务 4: 文本关键词推断
# ========================================================================

print("\n" + "="*80)
print("📝 任务 4: 文本关键词推断")
print("="*80)

# has_weight_change 从 weight_change_description 推断
if all(c in df.columns for c in ['has_weight_change', 'weight_change_description']):
    # 找到has_weight_change缺失但description存在的行
    if df['has_weight_change'].dtype == 'object':
        target_missing = df['has_weight_change'].isna() | (df['has_weight_change'] == '')
    else:
        target_missing = df['has_weight_change'].isna()
    
    desc_exists = df['weight_change_description'].notna() & (df['weight_change_description'] != '')
    to_infer = target_missing & desc_exists
    
    # 关键词匹配 - 在整个df上操作，然后用to_infer筛选
    desc_lower = df['weight_change_description'].astype(str).str.lower()
    
    # 有变化的关键词
    has_change = desc_lower.str.contains('decreased|decrease|下降|减轻|loss', na=False)
    no_change = desc_lower.str.contains('no_significant|无变化|正常', na=False)
    
    # 填充
    fill_yes = to_infer & has_change
    fill_no = to_infer & no_change
    
    df.loc[fill_yes, 'has_weight_change'] = 1
    df.loc[fill_no, 'has_weight_change'] = 0
    
    filled_count = fill_yes.sum() + fill_no.sum()
    if filled_count > 0:
        log_imputation('has_weight_change', 'Keyword Matching', 
                      'weight_change_description keywords', filled_count)

# ========================================================================
# 补全任务 5: 单位提取与转换
# ========================================================================

print("\n" + "="*80)
print("📝 任务 5: 单位提取与数值转换")
print("="*80)

# 血糖值提取 (去除单位)
if 'fasting_glucose' in df.columns:
    def extract_glucose(val):
        if pd.isna(val) or str(val) in ['', 'None', 'nan']:
            return np.nan
        match = re.match(r'(\d+\.?\d*)\s*mmol/L?', str(val), re.IGNORECASE)
        if match:
            return float(match.group(1))
        try:
            return float(val)
        except:
            return np.nan
    
    before_missing = df['fasting_glucose'].isna().sum()
    df['fasting_glucose_mmol'] = df['fasting_glucose'].apply(extract_glucose)
    after_valid = df['fasting_glucose_mmol'].notna().sum()
    filled_count = after_valid - (len(df) - before_missing)
    
    if filled_count > 0:
        log_imputation('fasting_glucose_mmol', 'Unit Extraction', 
                      'fasting_glucose parse mmol/L', filled_count)

# HbA1c提取
if 'hba1c' in df.columns:
    def extract_hba1c(val):
        if pd.isna(val) or str(val) in ['', 'None', 'nan']:
            return np.nan
        match = re.match(r'(\d+\.?\d*)\s*%?', str(val))
        if match:
            return float(match.group(1))
        try:
            return float(val)
        except:
            return np.nan
    
    df['hba1c_percent'] = df['hba1c'].apply(extract_hba1c)
    filled_count = df['hba1c_percent'].notna().sum()
    if filled_count > 0:
        log_imputation('hba1c_percent', 'Unit Extraction', 
                      'hba1c parse %', filled_count)

# ========================================================================
# 统计补全效果
# ========================================================================

print("\n" + "="*80)
print("📊 Tier 1 补全效果统计")
print("="*80)

# 计算补全后统计
final_stats = {}
for col in df.columns:
    missing_count, missing_rate = calc_missing_rate(df[col])
    final_stats[col] = {'count': missing_count, 'rate': missing_rate}

# 对比
print("\n【整体数据质量变化】")
initial_total_missing = sum(s['count'] for s in initial_stats.values())
final_total_missing = sum(s['count'] for s in final_stats.values())
improvement = initial_total_missing - final_total_missing

print(f"补全前总缺失: {initial_total_missing:,} 单元格")
print(f"补全后总缺失: {final_total_missing:,} 单元格")
print(f"绝对减少: {improvement:,} 单元格 ({improvement/initial_total_missing*100:.2f}%)")

# 按模块统计
print("\n【新增特征统计】")
new_features = [col for col in df.columns if col not in initial_stats.keys()]
for feat in new_features:
    valid_count = df[feat].notna().sum()
    print(f"  • {feat}: {valid_count} 有效值 ({valid_count/len(df)*100:.1f}%)")

# 原有特征改善
print("\n【原有特征改善 Top 10】")
improvements = []
for col in initial_stats.keys():
    if col in final_stats:
        initial_missing = initial_stats[col]['count']
        final_missing = final_stats[col]['count']
        if initial_missing > 0:
            reduced = initial_missing - final_missing
            reduction_pct = reduced / initial_missing * 100
            if reduced > 0:
                improvements.append((col, reduced, reduction_pct))

improvements.sort(key=lambda x: x[1], reverse=True)
for col, reduced, pct in improvements[:10]:
    print(f"  • {col}: 减少 {reduced} 条 ({pct:.1f}%)")

# ========================================================================
# 保存结果
# ========================================================================

print("\n" + "="*80)
print("💾 保存补全结果")
print("="*80)

# 保存补全后的数据集
output_path = '/Users/fanganita/.openclaw/workspace/胰腺癌数据_Tier1补全后.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 补全后数据集: {output_path}")

# 保存补全日志
log_df = pd.DataFrame(imputation_log)
log_path = '/Users/fanganita/.openclaw/workspace/Tier1补全日志.csv'
log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
print(f"✅ 补全操作日志: {log_path}")

# 生成补全报告
report = f"""
【Tier 1 规则补全实施报告】

实施时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
原始数据: {len(df)} 行 × {len(df.columns)} 列

补全操作汇总:
"""

for log in imputation_log:
    report += f"\n  • {log['feature']}\n"
    report += f"    方法: {log['method']}\n"
    report += f"    条件: {log['condition']}\n"
    report += f"    补全数: {log['count']} 条\n"
    if log['example']:
        report += f"    示例: {log['example']}\n"

report += f"""

效果统计:
  • 补全操作总数: {len(imputation_log)} 项
  • 总单元格缺失减少: {improvement:,} ({improvement/initial_total_missing*100:.2f}%)
  • 新增派生特征: {len(new_features)} 个

下一阶建议:
  • Tier 2: 使用LLM进行症状关联推断和临床分期补全
  • Tier 3: 使用LLM进行合并症风险估计
  • 验证: 与临床专家对比关键指标的合理性
"""

print(report)

report_path = '/Users/fanganita/.openclaw/workspace/Tier1补全报告.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✅ 补全报告: {report_path}")

print("\n" + "="*80)
print("✨ Tier 1 补全完成！")
print("="*80)
