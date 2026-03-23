#!/usr/bin/env python3
"""
胰腺癌数据集 - LLM驱动数据补全策略设计
策略文档 + 代码框架

核心设计原则：
1. 规则优先：简单逻辑用规则，降低成本
2. LLM增强：复杂医学推理用LLM
3. 分层补全：先基础特征，再依赖特征
4. 交叉验证：多源信息互相验证
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv('/Users/fanganita/.openclaw/media/inbound/pancreatic_cancer_data_normalized_clean---f3fc1dae-f432-4e85-a92d-62263d2ad843')

print("="*80)
print("🧠 LLM驱动数据补全策略设计")
print("="*80)

# ========================================================================
# 第一部分：特征分类与补全策略映射
# ========================================================================

imputation_strategy = {
    " Tier 1 - 规则补全 (零成本)": {
        "description": "基于确定性规则的逻辑补全，无需LLM",
        "features": {
            "duration_parsing": {
                "target_cols": [
                    "hepatitis_b_duration", "alcoholic_liver_duration", "diabetes_duration",
                    "smoking_duration", "drinking_duration", "pancreatic_cancer_duration"
                ],
                "text_cols": [
                    "hepatitis_b_duration_text", "alcoholic_liver_duration_text",
                    "diabetes_duration_text", "smoking_duration_text", "drinking_duration_text",
                    "pancreatic_cancer_duration_text"
                ],
                "method": "Regex + 单位换算",
                "logic": """
                解析文本格式：
                - "360.0_years" → 360.0 (年)
                - "60.0_years" → 60.0 (年) 
                - "4.0_months" → 4.0/12 = 0.33 (年)
                - "2.0_months" → 2.0/12 = 0.17 (年)
                - 统一转换为：duration_months (月) 或 duration_years (年)
                """,
                "code_template": """
                def parse_duration(text_value):
                    if pd.isna(text_value) or text_value == '0.0_units':
                        return np.nan
                    match = re.match(r'(\d+\.?\d*)_(years|months|days)', str(text_value))
                    if match:
                        value, unit = float(match.group(1)), match.group(2)
                        if unit == 'years':
                            return value * 12  # 转换为月
                        elif unit == 'months':
                            return value
                        elif unit == 'days':
                            return value / 30
                    return np.nan
                """
            },
            
            "conditional_zero": {
                "target_cols": [
                    "daily_smoking_amount", "smoking_duration", "quit_smoking_date",
                    "daily_drinking_amount", "drinking_duration", "quit_drinking_date",
                    "daily_smoking_cigarettes", "daily_alcohol_g"
                ],
                "condition_col": ["is_smoker", "is_drinker", "has_quit_smoking", "has_quit_drinking"],
                "method": "条件规则推断",
                "logic": """
                确定性补全规则：
                - is_smoker = 0 → daily_smoking_amount = 0, smoking_duration = 0
                - has_quit_smoking = 0 → quit_smoking_date = NaT (未戒烟)
                - is_drinker = 0 → daily_drinking_amount = 0, drinking_duration = 0
                - has_quit_drinking = 0 → quit_drinking_date = NaT
                """
            },
            
            "derived_calculation": {
                "target_cols": ["bmi", "pancreatic_cancer_duration_months"],
                "source_cols": {
                    "bmi": ["weight_kg", "height_cm"],
                    "pancreatic_cancer_duration_months": ["diagnosis_date", "admission_date"]
                },
                "method": "数学公式计算",
                "logic": """
                - BMI = weight_kg / (height_cm/100)^2
                - duration_months = (admission_date - diagnosis_date).days / 30
                """
            },
            
            "binary_from_text": {
                "target_cols": [
                    "has_weight_change"
                ],
                "source_cols": ["weight_change_description"],
                "method": "文本关键词匹配",
                "logic": """
                - weight_change_description 包含 'Decreased', '下降', '减轻' → has_weight_change = 1
                - weight_change_description = 'No_significant_change' → has_weight_change = 0
                """
            }
        }
    },
    
    " Tier 2 - LLM轻量补全 (低成本)": {
        "description": "基于单样本上下文的简单推断，单次LLM调用",
        "features": {
            "clinical_stage_inference": {
                "target_cols": ["clinical_stage", "clinical_t_stage", "clinical_n_stage", "clinical_m_stage"],
                "context_cols": [
                    "has_metastasis", "metastasis_site", "metastasis_to_*",
                    "surgery_method", "pathological_stage", "tb_before_treatment"
                ],
                "prompt_template": """
                基于以下患者信息，推断临床分期 (TNM分期)：
                
                患者信息：
                - 转移情况: {has_metastasis}, 转移部位: {metastasis_site}
                - 手术方式: {surgery_method}
                - 病理分期: {pathological_stage}
                - 胆红素水平: {tb_before_treatment}
                
                请输出：
                - clinical_stage: (如 Stage_I, Stage_II, Stage_III, Stage_IV)
                - clinical_t_stage: (T1-T4, X)
                - clinical_n_stage: (N0-N2, X)  
                - clinical_m_stage: (M0-M1, X)
                
                仅输出JSON格式。
                """,
                "validation": "与病理分期对比一致性"
            },
            
            "symptom_completeness": {
                "target_cols": [
                    "has_nausea", "has_vomiting", "has_diarrhea", 
                    "has_back_pain", "has_peptic_ulcer"
                ],
                "context_cols": [
                    "has_abdominal_pain", "has_abdominal_distension", "has_jaundice"
                ],
                "prompt_template": """
                胰腺癌患者症状推断：
                
                已知症状: {known_symptoms}
                
                请根据胰腺癌典型临床表现，推断以下症状存在的可能性(0-1)：
                - has_nausea (恶心)
                - has_vomiting (呕吐)
                - has_diarrhea (腹泻)
                - has_back_pain (背痛)
                - has_peptic_ulcer (消化性溃疡)
                
                考虑：黄疸患者常伴恶心；腹痛患者可能有背痛等。
                输出JSON格式，包含概率值和置信度。
                """
            },
            
            "family_history_details": {
                "target_cols": ["family_pancreatic_cancer_relation"],
                "condition_col": "family_pancreatic_cancer",
                "prompt_template": """
                患者有胰腺癌家族史 (family_pancreatic_cancer=1)。
                
                请推断最可能的亲属关系（一级亲属/二级亲属）：
                - 一级亲属：父母、兄弟姐妹、子女
                - 二级亲属：祖父母、叔伯姑姨
                
                输出：'First_degree' 或 'Second_degree' 或 'Unknown'
                """
            }
        }
    },
    
    " Tier 3 - LLM深度推理 (高成本)": {
        "description": "需要医学知识推理和多源信息融合",
        "features": {
            "comorbidity_inference": {
                "target_cols": [
                    "has_hp_infection", "has_eb_virus", "has_coronary_disease",
                    "has_hyperlipidemia"
                ],
                "context_cols": [
                    "age", "gender", "has_diabetes", "has_hypertension",
                    "fasting_glucose", "family_other_cancer", "birthplace"
                ],
                "prompt_template": """
                基于患者基础信息，推断未记录的合并症风险：
                
                患者档案：
                - 人口学: {age}岁, {gender}, {birthplace}
                - 已知疾病: 糖尿病{has_diabetes}, 高血压{has_hypertension}
                - 实验室: 空腹血糖{fasting_glucose}
                - 家族史: {family_other_cancer}
                
                请推断以下疾病的患病风险(0-1)及置信度：
                - has_hp_infection (幽门螺杆菌感染，中国感染率约50%)
                - has_eb_virus (EB病毒感染)
                - has_coronary_disease (冠心病，与年龄、糖尿病相关)
                - has_hyperlipidemia (高脂血症，与糖尿病、肥胖相关)
                
                结合流行病学数据和患者个体特征进行推理。
                输出JSON格式。
                """,
                "rag_enhancement": True,
                "rag_sources": ["中国慢性病流行病学数据", "胰腺癌风险因素研究"]
            },
            
            "cause_of_death_classification": {
                "target_cols": ["cause_of_death"],
                "condition": "is_deceased = 1",
                "context_cols": [
                    "has_recurrence", "has_metastasis", "metastasis_site",
                    "survival_days", "surgery_method", "stage_at_diagnosis"
                ],
                "prompt_template": """
                胰腺癌患者死亡原因推断：
                
                患者结局信息：
                - 生存时间: {survival_days}天
                - 复发: {has_recurrence}, 转移: {has_metastasis}
                - 转移部位: {metastasis_site}
                - 治疗方式: {surgery_method}
                - 诊断分期: {stage_at_diagnosis}
                
                请推断最可能的死亡原因：
                - Pancreatic_cancer_related (胰腺癌直接相关)
                - Treatment_complications (治疗并发症)
                - Other_cancer (其他恶性肿瘤)
                - Non_cancer_cause (非肿瘤原因)
                
                输出分类和置信度。
                """
            },
            
            "treatment_intention_reconstruction": {
                "target_cols": ["surgery_method", "surgery_name"],
                "context_cols": [
                    "stage_at_diagnosis", "has_metastasis", "age", "ecog_score",
                    "tb_before_treatment", "diagnosis_date", "surgery_date"
                ],
                "prompt_template": """
                基于患者临床特征，推断最适合的手术策略：
                
                患者情况：
                - 分期: {stage_at_diastole}, 转移: {has_metastasis}
                - 年龄: {age}, ECOG评分: {ecog_score}
                - 胆红素: {tb_before_treatment}
                - 诊断日期: {diagnosis_date}, 手术日期: {surgery_date}
                
                请推断：
                1. 手术意图：根治性切除 / 姑息性手术 / 减瘤手术
                2. 具体术式：Whipple / 远端胰腺切除 / 全胰腺切除 / 胆肠吻合等
                3. 是否联合门静脉切除等扩大清扫
                
                基于NCCN指南和患者具体情况推理。
                输出详细手术方案建议。
                """
            }
        }
    },
    
    " Tier 4 - 多智能体验证 (最高成本)": {
        "description": "多LLM协作 + 一致性验证 + 人工审核",
        "features": {
            "critical_value_verification": {
                "target_cols": [
                    "survival_days", "is_deceased", "has_recurrence"
                ],
                "method": "Multi-Agent Consensus",
                "workflow": """
                1. Agent 1 (时间线验证): 检查诊断-治疗-随访时间线合理性
                2. Agent 2 (医学逻辑): 验证结局与临床特征的一致性
                3. Agent 3 (数据校验): 检查数值范围合理性
                4. 投票机制：3个Agent中至少2个一致才接受
                5. 冲突案例标记为人工审核
                """
            }
        }
    }
}

# ========================================================================
# 第二部分：补全优先级与依赖图
# ========================================================================

print("\n" + "="*80)
print("📋 补全策略分层架构")
print("="*80)

for tier, content in imputation_strategy.items():
    print(f"\n{tier}")
    print(f"  描述: {content['description']}")
    if 'features' in content:
        print(f"  包含特征组:")
        for feat_group, details in content['features'].items():
            target_count = len(details.get('target_cols', []))
            print(f"    - {feat_group}: {target_count}个特征")

# ========================================================================
# 第三部分：核心补全算法代码框架
# ========================================================================

print("\n" + "="*80)
print("💻 核心补全代码框架")
print("="*80)

code_framework = '''
"""
LLM驱动数据补全系统 - 核心代码框架
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re
from datetime import datetime
import openai  # 或其他LLM API

class MedicalDataImputer:
    """医疗数据智能补全系统"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.imputation_log = []
        
    # ==================== Tier 1: 规则补全 ====================
    
    def parse_duration_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析持续时间文本"""
        duration_cols = [
            'hepatitis_b_duration', 'alcoholic_liver_duration', 'diabetes_duration',
            'smoking_duration', 'drinking_duration', 'pancreatic_cancer_duration'
        ]
        
        def extract_months(val):
            if pd.isna(val) or str(val) in ['0.0_units', '0', '']:
                return np.nan
            match = re.match(r'(\\d+\\.?\\d*)_(years|months|days)', str(val))
            if match:
                value, unit = float(match.group(1)), match.group(2)
                converters = {'years': 12, 'months': 1, 'days': 1/30}
                return value * converters.get(unit, 1)
            return np.nan
        
        for col in duration_cols:
            if col in df.columns:
                df[f'{col}_months'] = df[col].apply(extract_months)
                self.log_imputation(col, 'regex_parsing', 'Rule-based')
        
        return df
    
    def conditional_zero_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """条件零值补全"""
        rules = [
            ('is_smoker', 0, ['daily_smoking_amount', 'smoking_duration', 
                             'daily_smoking_cigarettes']),
            ('is_drinker', 0, ['daily_drinking_amount', 'drinking_duration',
                              'daily_alcohol_g']),
            ('has_quit_smoking', 0, ['quit_smoking_date']),
            ('has_quit_drinking', 0, ['quit_drinking_date'])
        ]
        
        for condition_col, condition_val, target_cols in rules:
            if condition_col in df.columns:
                mask = df[condition_col] == condition_val
                for target in target_cols:
                    if target in df.columns:
                        filled_count = df.loc[mask, target].isna().sum()
                        df.loc[mask, target] = 0 if 'date' not in target else pd.NaT
                        if filled_count > 0:
                            self.log_imputation(target, f'{condition_col}={condition_val}', 
                                              'Rule-based', filled_count)
        return df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生特征"""
        # BMI计算
        if all(c in df.columns for c in ['weight_kg', 'height_cm']):
            mask = df['bmi'].isna() & df['weight_kg'].notna() & df['height_cm'].notna()
            df.loc[mask, 'bmi'] = df.loc[mask, 'weight_kg'] / (df.loc[mask, 'height_cm']/100)**2
            self.log_imputation('bmi', 'weight+height', 'Formula', mask.sum())
        
        # 病程计算
        if all(c in df.columns for c in ['admission_date', 'diagnosis_date']):
            df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
            df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
            mask = df['pancreatic_cancer_duration_months'].isna()
            duration = (df.loc[mask, 'admission_date'] - df.loc[mask, 'diagnosis_date']).dt.days / 30
            df.loc[mask, 'pancreatic_cancer_duration_months'] = duration
            self.log_imputation('duration_months', 'date_diff', 'Formula', mask.sum())
        
        return df
    
    # ==================== Tier 2-3: LLM补全 ====================
    
    def llm_impute_single(self, row: pd.Series, feature: str, 
                          context_features: List[str], 
                          prompt_template: str) -> Dict:
        """单样本LLM补全"""
        
        # 构建上下文
        context = {f: row.get(f, 'Unknown') for f in context_features}
        prompt = prompt_template.format(**context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical data imputation assistant. Respond only in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            result = response.choices[0].message.content
            # 解析JSON结果
            import json
            parsed = json.loads(result)
            
            return {
                'feature': feature,
                'value': parsed.get(feature),
                'confidence': parsed.get('confidence', 0.5),
                'method': 'LLM',
                'cost': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {
                'feature': feature,
                'value': None,
                'confidence': 0,
                'method': 'LLM_FAILED',
                'error': str(e)
            }
    
    def batch_llm_impute(self, df: pd.DataFrame, feature: str,
                         condition_mask: pd.Series,
                         context_features: List[str],
                         prompt_template: str,
                         batch_size: int = 10) -> pd.DataFrame:
        """批量LLM补全（带缓存）"""
        
        to_impute = df[condition_mask].copy()
        
        # 简单的缓存机制
        cache_key = f"{feature}_{hash(prompt_template)}"
        
        results = []
        for idx, row in to_impute.iterrows():
            result = self.llm_impute_single(row, feature, context_features, prompt_template)
            results.append((idx, result))
            
            # 成本控制和进度显示
            if len(results) % batch_size == 0:
                print(f"  已处理 {len(results)}/{len(to_impute)} 条...")
        
        # 回填结果
        for idx, result in results:
            if result['value'] is not None:
                df.at[idx, feature] = result['value']
                self.log_imputation(feature, 'LLM', f"conf={result['confidence']:.2f}")
        
        return df
    
    # ==================== 验证与评估 ====================
    
    def cross_validate(self, df: pd.DataFrame, feature: str,
                       validation_rules: List[callable]) -> pd.DataFrame:
        """交叉验证补全结果"""
        
        df[f'{feature}_validated'] = True
        
        for rule in validation_rules:
            violations = rule(df)
            df.loc[violations, f'{feature}_validated'] = False
            
        invalid_count = (~df[f'{feature}_validated']).sum()
        print(f"  {feature}: {invalid_count} 个值未通过验证")
        
        return df
    
    def log_imputation(self, feature: str, condition: str, 
                       method: str, count: int = 1):
        """记录补全日志"""
        self.imputation_log.append({
            'timestamp': datetime.now(),
            'feature': feature,
            'condition': condition,
            'method': method,
            'count': count
        })
    
    def get_imputation_report(self) -> pd.DataFrame:
        """生成补全报告"""
        return pd.DataFrame(self.imputation_log)


# ==================== 使用示例 ====================

def main():
    """完整补全流程示例"""
    
    # 1. 加载数据
    df = pd.read_csv('pancreatic_cancer_data.csv')
    
    # 2. 初始化补全器
    imputer = MedicalDataImputer(api_key="your-api-key")
    
    # 3. Tier 1: 规则补全
    print("执行 Tier 1 - 规则补全...")
    df = imputer.parse_duration_text(df)
    df = imputer.conditional_zero_imputation(df)
    df = imputer.calculate_derived_features(df)
    
    # 4. Tier 2: LLM轻量补全
    print("执行 Tier 2 - LLM轻量补全...")
    # 例如：症状推断
    symptom_context = ['has_abdominal_pain', 'has_jaundice', 'has_vomiting']
    symptom_prompt = """基于症状{has_abdominal_pain}, {has_jaundice}, {has_vomiting}，
    推断患者是否有恶心(has_nausea)。输出JSON: {{"has_nausea": 0或1, "confidence": 0-1}}"""
    
    mask = df['has_nausea'].isna()
    df = imputer.batch_llm_impute(df, 'has_nausea', mask, 
                                  symptom_context, symptom_prompt)
    
    # 5. 生成报告
    report = imputer.get_imputation_report()
    print("\\n补全统计:")
    print(report.groupby(['method', 'feature'])['count'].sum())
    
    return df, report


if __name__ == "__main__":
    main()
'''

print(code_framework)

# ========================================================================
# 第四部分：LLM Prompt库
# ========================================================================

print("\n" + "="*80)
print("📝 LLM Prompt模板库")
print("="*80)

prompt_library = {
    "clinical_stage_inference": {
        "purpose": "推断临床分期",
        "prompt": """
你是一位资深的肿瘤学专家。请根据以下患者信息推断胰腺癌的临床分期(TNM分期)。

患者信息：
- 年龄: {age}岁
- 是否有转移: {has_metastasis}
- 转移部位: {metastasis_site}
- 手术方式: {surgery_method}
- 病理分期(如有): {pathological_stage}
- 总胆红素: {tb_before_treatment} μmol/L
- CA19-9水平(如有): 未提供

TNM分期规则：
- T1: 肿瘤局限在胰腺，≤2cm
- T2: 肿瘤局限在胰腺，>2cm  
- T3: 肿瘤侵犯十二指肠、胆管或胰周组织
- T4: 肿瘤侵犯大血管或周围器官
- N0: 无区域淋巴结转移
- N1: 1-3个区域淋巴结转移
- N2: ≥4个区域淋巴结转移
- M0: 无远处转移
- M1: 有远处转移

请输出JSON格式：
{{
    "clinical_stage": "Stage_I/II/III/IV/Unknown",
    "clinical_t_stage": "T1/T2/T3/T4/X",
    "clinical_n_stage": "N0/N1/N2/X", 
    "clinical_m_stage": "M0/M1/X",
    "confidence": 0.0-1.0,
    "reasoning": "简要推理过程"
}}
"""
    },
    
    "comorbidity_risk": {
        "purpose": "推断合并症风险",
        "prompt": """
基于患者的流行病学背景和已知疾病，推断以下合并症的患病概率。

患者信息：
- 年龄: {age}岁，性别: {gender}
- 地区: {birthplace}
- 已知疾病: 糖尿病{has_diabetes}, 高血压{has_hypertension}
- 血糖: {fasting_glucose} mmol/L
- 家族史: {family_other_cancer}

流行病学参考数据：
- 中国幽门螺杆菌感染率: ~50%
- EB病毒血清阳性率: >90%(成人)
- 冠心病患病率(>60岁): 约10-15%
- 高脂血症患病率(糖尿病患): >60%

请输出JSON：
{{
    "has_hp_infection": {{"probability": 0.0-1.0, "confidence": 0.0-1.0}},
    "has_eb_virus": {{"probability": 0.0-1.0, "confidence": 0.0-1.0}},
    "has_coronary_disease": {{"probability": 0.0-1.0, "confidence": 0.0-1.0}},
    "has_hyperlipidemia": {{"probability": 0.0-1.0, "confidence": 0.0-1.0}}
}}
"""
    },
    
    "survival_outcome_verification": {
        "purpose": "验证生存结局一致性",
        "prompt": """
验证以下患者生存结局的合理性：

患者信息：
- 生存天数: {survival_days}
- 是否死亡: {is_deceased}
- 是否复发: {has_recurrence}
- 诊断分期: {stage_at_diagnosis}
- 是否手术: {surgery_method}
- 转移情况: {has_metastasis}, 部位: {metastasis_site}

胰腺癌典型生存期：
- I期(手术): 中位生存 20-30月
- II期(手术): 中位生存 15-20月
- III期: 中位生存 8-12月
- IV期: 中位生存 3-6月

请判断：
1. 生存时间是否与分期一致？(consistent/inconsistent/uncertain)
2. 如有明显异常，指出可能的解释
3. 建议的修正(如有)

输出JSON格式。
"""
    }
}

for name, content in prompt_library.items():
    print(f"\n【{name}】")
    print(f"用途: {content['purpose']}")
    print(f"Prompt长度: {len(content['prompt'])} 字符")

# ========================================================================
# 第五部分：补全效果评估指标
# ========================================================================

print("\n" + "="*80)
print("📊 补全效果评估指标体系")
print("="*80)

evaluation_metrics = """
## 1. 补全覆盖率指标

| 指标 | 公式 | 目标值 |
|------|------|--------|
| 特征级覆盖率 | 已补全特征数 / 可补全特征数 | >80% |
| 样本级覆盖率 | 至少补全1项的样本数 / 总样本数 | >90% |
| 总体缺失率下降 | (补全前缺失率 - 补全后缺失率) / 补全前缺失率 | >50% |

## 2. 补全准确性指标（需金标准验证）

| 指标 | 说明 | 计算方法 |
|------|------|----------|
| 分类特征准确率 | 离散值的正确率 | Accuracy / F1 |
| 连续特征MAE | 连续值的平均绝对误差 | mean(\|预测-真实\|) |
| 排名一致性 | 序关系的保持程度 | Kendall's Tau |
| 分布相似度 | 补全后 vs 原始分布 | KS检验 / Wasserstein距离 |

## 3. 下游任务验证指标

| 任务 | 评估指标 | 验证方法 |
|------|----------|----------|
| 生存预测 | C-index | 补全前 vs 补全后模型性能 |
| 转移风险 | AUC-ROC | 同上 |
| 治疗方案推荐 | 临床专家一致性 | 专家评审 |

## 4. 文献对照验证

- 关键流行病学指标对比（如：性别比例、中位生存期）
- 与已发表研究的一致性检验
- 异常值识别和处理

## 5. 成本效益分析

- Token消耗统计
- API调用成本
- 补全每特征的平均成本
- 与人工标注成本对比
"""

print(evaluation_metrics)

# 保存完整策略文档
print("\n" + "="*80)
print("💾 保存策略文档")
print("="*80)

# 将策略导出为JSON
import json
strategy_output = {
    "metadata": {
        "dataset": "Pancreatic Cancer EMR",
        "total_features": len(df.columns),
        "total_samples": len(df),
        "avg_missing_rate": 68.0,
        "created": datetime.now().isoformat()
    },
    "strategy": imputation_strategy,
    "prompts": prompt_library,
    "evaluation": evaluation_metrics
}

with open('/Users/fanganita/.openclaw/workspace/LLM补全策略_完整方案.json', 'w', encoding='utf-8') as f:
    json.dump(strategy_output, f, ensure_ascii=False, indent=2)

# 保存代码框架
with open('/Users/fanganita/.openclaw/workspace/LLM补全系统_代码框架.py', 'w', encoding='utf-8') as f:
    f.write(code_framework)

print("✅ 策略JSON: LLM补全策略_完整方案.json")
print("✅ 代码框架: LLM补全系统_代码框架.py")
print("\n策略设计完成！")
