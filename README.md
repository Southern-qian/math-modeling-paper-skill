# math-modeling-paper — 数模国奖论文生产技能

将"赛题 PDF → 真实计算 → 文献核验 → LaTeX 排版 → QA 验收 → 评分交付"的完整论文生产线封装为一个可复用技能。实战验证：2025 高教社杯全国大学生数学建模竞赛 B 题《碳化硅外延层厚度的确定》，成品 23 页 / 15 图 / 6 表，三方法互证 <1%，与开源社区六种独立实现系统对照，judge 视觉验收通过，自评 ≈91.5/100（国一门槛）。

## 目录结构

```
math-modeling-paper/
├── SKILL.md                  # 技能主文件：九阶段工作流总览 + 8 条硬规则
├── references/
│   ├── workflow.md           # 九阶段详细操作手册（含命令与代码片段）
│   ├── pitfalls.md           # 24 条实战踩坑清单（量纲/LaTeX/数值一致性/流程）
│   ├── gpt-guide.md          # GPT 版写文思路与技巧（可整段粘贴给任意 LLM）
│   └── self-check.md         # 人工验收清单（25 条，A/B/C 三档 + 评分表）
└── assets/
    └── paper-template.tex    # 国赛格式 LaTeX 骨架（Tectonic 直接编译）
```

## 两种用法

**用法 A：ZCode 技能** — 把整个目录放到 `~/.agents/skills/math-modeling-paper`，ZCode 会在你说"写数模论文/做 X 题完整解答"时自动加载。

**用法 B：任意 LLM（GPT/Claude 等）** — 打开 `references/gpt-guide.md`，把【系统提示】整段粘贴给模型，再按文末格式提供赛题与数据。核心纪律：**分阶段推进，数值表确认前不许写正文；每阶段输出五段式交接摘要；tex+图打包走 Overleaf（XeLaTeX）编译；成品按 `self-check.md` 25 条清单人工验收**。

## 核心理念

1. 每个数值来自对附件数据的真实计算（代码可复现，结果存 JSON）；
2. 每条参考文献过 Crossref 核验，核验不过直接删；
3. 主方法 + 两个独立交叉验证，互差 <2%；
4. 至少一个有机理分析的改进点 + 一个可辨识实验；
5. 可靠性四件套：双数据互证、Bootstrap、灵敏度表、蒙特卡洛；
6. 正文-表格数字互查，每个百分比可复算；
7. 全页渲染视觉验收循环直至 pass；
8. 七维 rubric 评分 + 距国一差距清单。

## 相关仓库（实战案例）

- 原实战论文与全部复现代码：见交付目录（final.pdf / main_paper_v2.tex / analysis*.py）
- 2025 高教社杯 B 题开源实现调研对象：jiebro0721/cumcm-2025b-sic-epitaxy、SchrodingerJia/SiC_Thickness_Analysis、Skyler-Luo/CUMCM2025-B、LinJK1/sic-epitaxial-thickness-inversion、FWSY0623/SiC-Epitaxial-Thickness-Measurement-via-IR-Interference、cc1107yss/cumcm-2025-sic-infrared-thickness
