# 九阶段详细操作手册

以 2025 高教社杯 B 题（碳化硅外延层厚度，物理光学类）为实例路径。其他题型（优化/统计/机器学习类）替换第 4 阶段的模型内核即可，其余七个阶段完全同构。

## 阶段 1：题目解析

```bash
python3 -c "import pymupdf; doc=pymupdf.open('B题.pdf'); [print(p.get_text()) for p in doc]"
```

- 提取后逐问列出：已知量、求解量、显式约束（"折射率不是常数"这类提示往往是评分点）、隐式要求（"分析可靠性""设法消除影响"都要求定量证据）。
- 记录附件说明：每个附件的物理对象、入射角、列含义、单位。

## 阶段 2：数据勘察

```python
import pandas as pd
df = pd.read_excel('附件1.xlsx', header=0)
print(df.shape, df.columns.tolist())
print(df.head(3), df.tail(2))
```

必查四项：采样区间与步长（决定 FFT 分辨率）、量纲（波数 cm⁻¹ vs 波长 nm）、首尾异常值（边缘伪点）、物理特征波段（如剩余射线带：条纹消失区，需剔除并给出物理解释）。

## 阶段 3：案例与文献调研

**GitHub 开源挖掘**（GitHub 网页搜索限流时走 API）：

```
GET api.github.com/search/repositories?q=<赛题关键词>&per_page=15
GET api.github.com/repos/<owner>/<repo>/readme          # base64 → utf-8
GET api.github.com/repos/<owner>/<repo>/git/trees/main?recursive=1
GET api.github.com/repos/<owner>/<repo>/contents/<URL-encoded 中文路径>
```

按价值排序阅读：① 声称对齐命题人讲评的仓库（金标准）；② 带获奖论文 PDF 的仓库；③ 方法有独到之处的仓库（如 FFT 二次谐波判据）。记录每个仓库的：物理模型、反演算法、报告数值——这些进论文的"开源对照表"，也是你方法改进的靶子。

**文献真实性核验**（写论文前完成，不要拖到参考文献阶段）：

```python
import urllib.request, json
# 按 DOI 精确核验
json.load(urllib.request.urlopen("https://api.crossref.org/works/10.1007/s11664-998-0404-9"))
# 按标题模糊检索
json.load(urllib.request.urlopen("https://api.crossref.org/works?query.bibliographic=<标题>&rows=3"))
```

目标文献骨架（光学测厚类）：行业标准（ASTM F95）→ 开山经典（Phys. Rev. 1957/1959）→ 材料光学常数（Laser & Photonics Rev / Surf. Sci. Spectra）→ 最新进展（Elsevier/SPIE 近两年）→ 教材（Born & Wolf、Yeh）→ 开源实现 [EB/OL]。核验不过的条目直接弃用。

## 阶段 4：真实计算（核心）

组织成一个可重跑的 `analysis.py`，结构：

```
常量与物理模型函数（色散、菲涅耳、Airy）
  ↓ 预处理函数（截段、SG 滤波、本底）
  ↓ 特征提取（极值检测+剪枝、级数标定回归、FFT）
  ↓ 问题一/二/三各自的求解段（打印关键数值）
  ↓ 可靠性段（bootstrap、灵敏度扰动、蒙特卡洛）
  ↓ 图表段（全部 savefig 200dpi）
  ↓ json.dump(results)
```

**方法体系设计原则**（国奖评审最看重）：
- 主方法 + 两个独立交叉验证方法，三者数值互差 <2%；
- 至少一个"改进点"：对基线方法的已知缺陷给出机理分析（如：色散下极值间距由群指数 n_g=n+σ·dn/dσ 而非相速折射率决定）并用精确公式消除；
- 至少一个"可辨识实验"：扫描某个物理参数（如振子强度 ε∞），画目标函数曲线证明参数可辨识；
- 定性判定升级为定量判据（如：多光束干涉 → FFT 二次谐波 SNR 与相对强度 R_rel）。

**可靠性四件套**：双角度（双组数据）互证、Bootstrap 置信区间、参数扰动灵敏度表、蒙特卡洛噪声实验（30 次，报中位数与 MAD 而非均值±std）。

## 阶段 5：科研配图

```python
plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
# 每图：figsize=(7.2,3.5), dpi=200, 上/右脊柱隐藏, 虚线网格 alpha=0.35, legend 无框 fontsize=9
```

15 张图的黄金配置（物理类论文）：原始谱+滤波 | 极值标注 | 参数扫描曲线 | 回归/拟合优度图 | 机理对比曲线（相速 vs 群指数）| 频域判据图 | 全谱拟合对比 | 残差对比 | 物理参数曲线（Drude）| 优化收敛曲线 | 厚度分布直方图 | 蒙特卡洛分布 | 与开源实现对比条形图 | 折射率多方法对比 | TikZ 光路图与算法流程图。

## 阶段 6：LaTeX 生产

国赛格式要点（区别于普通学术模板）：
- `ctexart` + `fontset=fandol`（Tectonic 可用）；**无封面无目录**，第 1 页 = 居中标题 + "摘 要" + 分问题摘要（每段加粗"针对问题X"开头，每个结论带数值）+ 关键词；
- 章节阿拉伯数字；依次：问题重述 / 问题背景与研究现状 / 问题分析（含候选方案对比表）/ 模型假设（每条带依据）/ 符号说明 / 各问题模型建立与求解 / 与开源实现对照 / 灵敏度与稳健性 / 模型评价与推广 / 参考文献 / 代码附录；
- 引用：`\usepackage[numbers,super,sort&compress]{natbib}` + 手写 `thebibliography`（GB/T 7714 条目）；
- 表：booktabs 三线表；宽表用 tabularx + 自定义 `\newcolumntype{Y}{>{\raggedright\arraybackslash}X}` 防 underfull；
- URL 断行：`\usepackage{xurl}`；hyperref 最后加载并写全 `\hypersetup` 元数据。

编译：`tectonic main.tex` 两遍（包装脚本若有 2 分钟超时限制，先直接跑一遍预热宏包缓存）。字体用 fandol 最稳。

## 阶段 7：QA 流水线

1. `check-tex`：ASCII 引号、表格溢出、图宽；
2. Tectonic 两遍编译：**Overfull/Underfull 必须清零**（表格列加 raggedright、长公式拆行、TikZ 累积宽度用 inner sep 收缩）；
3. `font.check`：无豆腐块；
4. `pdf_qa`：元数据、边距对称、无空白页（公式溢出告警多为启发式误报，以编译日志为准）；
5. **数值一致性互查**（人工/脚本）：摘要 = 正文 = 表图；每个百分比从表内数据复算。

## 阶段 8：视觉验收循环

```python
import pymupdf; doc = pymupdf.open("final.pdf")
for i, p in enumerate(doc): p.get_pixmap(dpi=110).save(f"pages/p{i+1:02d}.png")
```

将全部页面 PNG 交给视觉审稿 agent，审查清单：中文渲染、图内中文标签、图题编号与正文引用一致、公式溢出、三线表完整、TikZ 标注与公式定义逐项一致、参考文献格式、代码框溢出、空节/孤行、浮动体落页。产出 JSON 判定 → 修复 must_fix → **只重渲染改动页并复审**（全量重跑浪费且会让 judge 看到陈旧 PNG 混淆）。若正文改动导致版面后移，必须全量重渲染后再复审。

## 阶段 9：评分与交付

按七个维度评分（满分 100）：摘要 15 / 假设与符号 10 / 模型与创造性 25 / 算法与结果 20 / 检验与灵敏度 10 / 表述与图表 10 / 学术规范 10。评分必须附"距国一的差距清单"（常见：篇幅与附录完整度、创新点的展示强度、摘要信息密度），让用户知道下一步往哪扩。
