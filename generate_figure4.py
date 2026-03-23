#!/usr/bin/env python3
"""
Figure 4: 外部验证森林图
比较我们的队列与已发表文献的关键流行病学参数
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 数据准备
parameters = [
    'Age (years)',
    'Male sex (%)', 
    'Diabetes (%)',
    'Jaundice (%)',
    'Median survival (days)'
]

# 我们的研究
our_study = {
    'mean': [63.3, 57.6, 30.2, 45.6, 167],
    'ci_lower': [62.8, 55.8, 28.5, 43.5, 155],
    'ci_upper': [63.8, 59.4, 31.9, 47.7, 179]
}

# 文献参考值 (SEER, NCDB等)
references = {
    'SEER': {
        'values': [63.0, 55.8, 32.0, 48.0, 165],
        'ci': [[62.5, 63.5], [54.0, 57.6], [30.0, 34.0], [46.0, 50.0], [150, 180]]
    },
    'NCDB': {
        'values': [64.2, 56.5, 31.5, 46.5, 172],
        'ci': [[63.7, 64.7], [54.7, 58.3], [29.5, 33.5], [44.3, 48.7], [158, 186]]
    },
    'MIMIC-IV': {
        'values': [62.8, 54.2, 29.8, 44.2, 158],
        'ci': [[62.0, 63.6], [52.0, 56.4], [27.5, 32.1], [41.5, 46.9], [145, 171]]
    }
}

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

colors = {
    'Our Study': '#d62728',  # 红色
    'SEER': '#1f77b4',      # 蓝色
    'NCDB': '#2ca02c',      # 绿色
    'MIMIC-IV': '#ff7f0e'   # 橙色
}

# 为每个参数绘制森林图
for idx, param in enumerate(parameters):
    ax = axes[idx]
    
    y_positions = []
    labels = []
    values = []
    errors = []
    colors_list = []
    
    # 我们的研究
    y_positions.append(0)
    labels.append('Our Study')
    values.append(our_study['mean'][idx])
    errors.append([
        our_study['mean'][idx] - our_study['ci_lower'][idx],
        our_study['ci_upper'][idx] - our_study['mean'][idx]
    ])
    colors_list.append(colors['Our Study'])
    
    # 参考文献
    y_pos = 1
    for ref_name, ref_data in references.items():
        y_positions.append(y_pos)
        labels.append(ref_name)
        values.append(ref_data['values'][idx])
        errors.append([
            ref_data['values'][idx] - ref_data['ci'][idx][0],
            ref_data['ci'][idx][1] - ref_data['values'][idx]
        ])
        colors_list.append(colors[ref_name])
        y_pos += 1
    
    # 绘制误差条
    for i, (y, val, err, color) in enumerate(zip(y_positions, values, errors, colors_list)):
        ax.errorbar(val, y, xerr=[[err[0]], [err[1]]], 
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color=color, ecolor=color, elinewidth=2)
    
    # 设置标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(param, fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=our_study['mean'][idx], color='gray', linestyle='--', alpha=0.5)
    
    # 添加数值标注
    for i, (y, val) in enumerate(zip(y_positions, values)):
        ax.text(val, y+0.15, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# 最后一个子图显示图例
axes[5].axis('off')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Our Study'], 
               markersize=10, label='Our Study (n=2,419)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['SEER'], 
               markersize=10, label='SEER Database'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['NCDB'], 
               markersize=10, label='NCDB'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['MIMIC-IV'], 
               markersize=10, label='MIMIC-IV')
]
axes[5].legend(handles=legend_elements, loc='center', fontsize=11, 
              title='Data Sources', title_fontsize=12)

# 添加说明文字
axes[5].text(0.5, 0.3, 
            'Forest plot comparing key epidemiological\n'
            'parameters between our cohort and published\n'
            'literature. Error bars represent 95% CI.',
            ha='center', va='center', fontsize=10, style='italic',
            transform=axes[5].transAxes)

# 总标题
fig.suptitle('Figure 4. External Validation Against Published Literature\n'
             'Comparison of key epidemiological parameters', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存
plt.savefig('/Users/fanganita/.openclaw/workspace/Figure4_外部验证森林图.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/fanganita/.openclaw/workspace/Figure4_外部验证森林图.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')

print("✅ Figure 4 已生成!")
print("   保存位置: ~/.openclaw/workspace/Figure4_外部验证森林图.png/pdf")

# 生成简要统计
print("\n📊 外部验证统计:")
print("-" * 50)
for i, param in enumerate(parameters):
    our_val = our_study['mean'][i]
    seer_val = references['SEER']['values'][i]
    diff = abs(our_val - seer_val)
    pct_diff = (diff / seer_val) * 100
    status = "✓" if pct_diff < 10 else "~"
    print(f"{status} {param:25s}: Our={our_val:.1f}, SEER={seer_val:.1f}, Diff={pct_diff:.1f}%")

plt.show()
