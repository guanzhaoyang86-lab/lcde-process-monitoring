"""Render the verified LC-DE study snapshot as a Chinese research PDF."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Flowable, KeepTogether,
)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "LCDE_模型与实验结果报告_2026-10-18.pdf"
RESULT_PATH = ROOT / "outputs/skab_frozen_baseline_comparison/comparison_result.json"
result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
frozen = json.loads((ROOT / "outputs/lcde_skab_v2_self_calibrated/frozen_method.json").read_text())
development = json.loads((ROOT / "outputs/lcde_skab_v2_self_calibrated/development_result.json").read_text())
A = result["aggregate"]
C = result["clean_commissioning_sensitivity"]["aggregate"]
pdfmetrics.registerFont(TTFont("CN", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("CNBold", "C:/Windows/Fonts/msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFontFamily("CN", normal="CN", bold="CNBold", italic="CN", boldItalic="CNBold")
NAVY = colors.HexColor("#142F43")
TEAL = colors.HexColor("#007E87")
GRAY = colors.HexColor("#506471")
LIGHT = colors.HexColor("#EEF5F6")
LINE = colors.HexColor("#D8E3E7")
INK = colors.HexColor("#203441")
ORANGE = colors.HexColor("#A45D2A")
W, H = A4
WIDTH = W - 92

STYLES = {
    "body": ParagraphStyle("body", fontName="CN", fontSize=10, leading=17,
        textColor=INK, spaceAfter=9, wordWrap="CJK"),
    "small": ParagraphStyle("small", fontName="CN", fontSize=8.5, leading=13.5,
        textColor=GRAY, spaceAfter=7, wordWrap="CJK"),
    "table": ParagraphStyle("table", fontName="CN", fontSize=8.2, leading=12,
        textColor=INK, wordWrap="CJK"),
    "th": ParagraphStyle("th", fontName="CNBold", fontSize=8.3, leading=12,
        textColor=colors.white, wordWrap="CJK"),
    "h1": ParagraphStyle("h1", fontName="CNBold", fontSize=22, leading=30,
        textColor=NAVY, spaceAfter=16, wordWrap="CJK"),
    "h2": ParagraphStyle("h2", fontName="CNBold", fontSize=12, leading=18,
        textColor=TEAL, spaceBefore=9, spaceAfter=7, wordWrap="CJK"),
    "formula": ParagraphStyle("formula", fontName="CN", fontSize=9.3, leading=16,
        textColor=NAVY, spaceAfter=5, wordWrap="CJK"),
    "title": ParagraphStyle("title", fontName="CNBold", fontSize=29, leading=39,
        textColor=NAVY, spaceAfter=10, wordWrap="CJK"),
}
story = []


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def add(text, style="body"):
    story.append(p(text, style))


def section(number, title, continuation=None):
    if story:
        story.append(PageBreak())
    add(f"{number}　{title}", "h1")
    if continuation:
        add(continuation, "small")


def sub(title):
    add(title, "h2")


def table(headers, rows, widths, highlights=(), size=8.2):
    style = ParagraphStyle("local_table", parent=STYLES["table"], fontSize=size)
    cells = [[p(str(h), "th") for h in headers]]
    cells += [[Paragraph(str(c), style) for c in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
    ]
    for i in range(1, len(cells)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F6F8F9")))
        if i - 1 in highlights:
            commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#DDF1EE")))
    t.setStyle(TableStyle(commands))
    story.extend([t, Spacer(1, 9)])


def box(title, text):
    t = Table([[p(title, "h2")], [p(text)]], colWidths=[WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([t, Spacer(1, 10)])


class Pipeline(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 172

    def draw(self):
        c = self.canv
        def cell(x, y, w, h, title, detail, fill=LIGHT):
            c.setFillColor(fill)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
            for txt, font, fs, dy, color in [(title,"CNBold",10, h-18,NAVY),
                                            (detail,"CN",8.4,h-33,GRAY)]:
                c.setFont(font,fs); c.setFillColor(color)
                c.drawCentredString(x+w/2,y+dy,txt)
        def arrow(x1,y1,x2,y2):
            c.setStrokeColor(TEAL);c.setLineWidth(1)
            c.line(x1,y1,x2,y2)
            c.line(x2,y2,x2-3,y2+5); c.line(x2,y2,x2+3,y2+5)
        cell(110,124,WIDTH-220,44,"8 维传感器序列","前 300 行初始化并标准化")
        cell(0,62,240,46,"观测统计分支：4 类证据","能量 / 均值 / 方差 / 差分能量")
        cell(WIDTH-240,62,240,46,"预测残差分支：1 类证据","健康 GRU 多步预测残差能量")
        c.setStrokeColor(TEAL)
        c.line(WIDTH/2,124,WIDTH/2,116)
        c.line(120,116,WIDTH-120,116)
        arrow(120,116,120,108); arrow(WIDTH-120,116,WIDTH-120,108)
        cell(48,0,WIDTH-96,43,"局部稳健归一化 → top-2 融合 → 有限窗口扫描","输出连续异常分数 S(t) 与阈值报警")
        c.line(120,62,120,52);c.line(WIDTH-120,62,WIDTH-120,52)
        c.line(120,52,WIDTH-120,52);arrow(WIDTH/2,52,WIDTH/2,43)


class ComparisonBars(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 145

    def draw(self):
        c=self.canv
        rows=[("LC-DE 完整模型","lcde_dual"),("LC-DE raw-only","lcde_raw_only"),
              ("PCA（全局）","global/pca_t2_plus_q"),
              ("预测残差（自校准）","self_starting/predictive_residual_cusum")]
        x0=153; bw=WIDTH-x0-36
        c.setFont("CN",8);c.setFillColor(GRAY)
        c.drawString(0,132,"统一监测区间的平均 AP；坐标从 0 开始")
        for i,(label,key) in enumerate(rows):
            y=108-i*27; val=A[key]["mean_postwarmup_ap"]
            c.setFont("CN",8.5);c.setFillColor(INK);c.drawString(0,y+3,label)
            c.setFillColor(LIGHT);c.rect(x0,y,bw,13,stroke=0,fill=1)
            c.setFillColor(TEAL if key=="lcde_dual" else colors.HexColor("#8AADB5"))
            c.rect(x0,y,bw*val,13,stroke=0,fill=1)
            c.setFillColor(INK);c.drawString(x0+bw+5,y+2,f"{val:.3f}")
        c.setFillColor(GRAY);c.setFont("CN",7.5)
        for tick in (0,.25,.5,.75,1):
            c.drawCentredString(x0+bw*tick,9,str(tick))


def footer(c, doc):
    c.saveState()
    c.setTitle("LC-DE：局部对照双能量顺序监测模型与实验结果")
    c.setAuthor("MATH 项目研究记录")
    c.setSubject("工业过程异常检测；SKAB；LC-DE；固定基线比较")
    c.setFillColor(TEAL);c.rect(46,H-33,25,3,fill=1,stroke=0)
    c.setFillColor(GRAY);c.setFont("CN",8)
    c.drawString(80,H-33,"LC-DE  /  模型与实验结果报告")
    c.setStrokeColor(LINE);c.line(46,40,W-46,40)
    c.setFont("CN",7.5);c.setFillColor(GRAY)
    c.drawString(46,27,"研究阶段结果  |  整理日期：2026-10-18  |  数据快照：2026-10-13")
    c.drawRightString(W-46,27,f"{doc.page:02d}")
    c.restoreState()


# Page 1: background, with an integrated title rather than an empty cover.
add("LC-DE", "title")
add("局部对照双能量顺序监测\n模型与实验结果".replace("\n","<br/>"), "title")
add("Local-Contrast Dual-Energy Scan · SKAB 实验报告", "small")
story.append(Spacer(1,10))
box("当前结论", "在同一组 16 条测试实验上，LC-DE 的平均 AP 为 <b>0.8266</b>，平均 F1 为 <b>0.4085</b>，故障持续期间检出 <b>9/16</b>。AP、F1 点估计高于本项目已运行的传统基线；但预测分支相对于 raw-only 消融的额外收益尚未得到充分证明。[4]")
add("01　背景", "h1")
add("工业设备的运行状态通常由压力、温度、流量、电流和振动等多路传感器共同反映。异常未必表现为某一变量的巨大偏移，也可能体现为局部波动增强、差分能量变化，或原有时序关系难以被健康模型预测。我们关注的是：仅利用健康数据拟合基础模型，在新的实验运行中持续输出异常证据。")
add("这一任务的难点在于正常运行之间也存在差异。固定的全局参照可能把正常工况差异当成故障；单纯的平方残差能量又可能遗漏原始观测中的结构变化。另一方面，长期累积的统计量容易保留很久以前的偏差，影响后续报警的可解释性。这些问题构成当前方法的设计动机，而非已经证明的普遍失效规律。")
sub("本阶段的研究重点")
add("本报告以可复核的模型定义、统一实验条件下的基线比较和消融分析为核心。工程部署指标不作为当前研究能否继续的硬性门槛；性能优劣仍需同时考虑排序质量、固定阈值下的检测能力、误报与延迟。")
add("阅读顺序：背景 → 介绍 → 我们的模型 → 使用的数据集 → 我们的结果。当前方法为 LC-DE；CRIS、EA-HX 等历史版本仅用于交代模型来源或充当对照，其结果不计入 LC-DE 的成绩。", "small")

# Page 2: introduction / definitions.
section("02","介绍")
sub("2.1 任务定义与方法定位")
add("设一条实验运行的观测为 x(t)∈R⁸。模型从初始化阶段结束后逐时产生分数 S(t)，并以固定阈值 h 给出报警 I[S(t)&gt;h]。输出是过程异常分数与报警状态；当前实现不承担故障类别诊断、根因定位或剩余寿命预测。")
add("预测骨干只用健康数据训练。局部对照统计量也由健康参照与运行前缀构造；但 24 组监测配置使用了开发集标签进行选择。因此，更准确的定位是<b>健康数据拟合、开发标签参与配置选择的异常监测方法</b>，而非整个流程完全不使用故障标签。")
sub("2.2 本报告采用的评价口径")
table(["指标","定义及解释"],[
    ["平均 AP ↑","每条实验计算非插值 Average Precision，再对实验等权平均；衡量异常分数排序质量，不是准确率，也不等同于任意插值 PR 曲线面积。"],
    ["平均 F1 ↑","在冻结阈值下逐条计算未做 point adjustment 的 F1，再等权平均。AP、F1 均只评估第 301 行起的公共监测区间。"],
    ["故障期间检出 ↑","必须出现位于 anomaly=1 区间内的新报警上升沿。故障开始前一直持续的报警，以及故障恢复后才产生的报警，均不计为该项检出。"],
    ["延迟中位数 ↓","仅对故障期间成功检出的运行，计算从真实故障起点至首个有效报警的时间；漏检不进入中位数，因此不能单独用于排名。"],
    ["故障前误报 ↓","统计首次故障开始前出现报警的实验条数。它不覆盖故障恢复后的误报，不能直接等同于整体零误报。"],
    ["健康样本误报率 ↓","逐条计算监测区间内 anomaly=0 样本的报警比例，再取平均；包括故障恢复后的健康样本。"],
], [108,WIDTH-108])
sub("2.3 公平比较的范围")
add("全部方法使用相同的传感器、开发/测试文件划分和前 300 行标准化。基线覆盖全局参照与运行自校准两种实现；阈值均取同一健康校准段上对应统计路径的最大值。比较前已精确复现两套基线的 18×8=144 条开发分数路径，最大绝对误差为 0。[4]")
add("该阈值规则统一，但不保证各方法拥有相同的实际误报率；本报告因此同时展示 AP 与误报。结果也不能直接与其他论文采用不同划分、阈值或 point adjustment 的数字横向比较。", "small")

# Page 3: model overview and predictor.
section("03","我们的模型","3.1 整体结构：LC-DE v2 自校准版本")
story.extend([Pipeline(),Spacer(1,8)])
add("LC-DE 将原始过程的统计变化与预测模型的残差变化并行提取，再将不同量纲的证据归一化并累积。当前模型的主要设计在于监测统计量与融合方式；预测骨干采用已有的多步概率 GRU 结构。[2,3]")
sub("3.2 预测分支：健康多步概率 GRU")
table(["项目","当前已训练配置"],[
    ["输入 / 网络","8 个传感器目标；兼容输入还含 1 个常零协变量、8 个全 1 mask 和 1 个默认零时间间隔特征，共 18 维。2 层单向 GRU，隐藏维度 64；44,208 个参数。"],
    ["历史 / 预测步长","历史上下文 60 个样本；预测步长为 1、5、15。线性输出头给出每个步长、每个目标的均值与尺度参数，共 48 维。"],
    ["训练 / 硬件","健康记录前 50% 用于模型拟合；40 epochs，batch size 1024，学习率 0.0003，weight decay 0.0001；RTX 5090 Laptop GPU。"],
    ["残差 / 用途","按预测步长对齐后，形成标准化创新残差 (T,3,8)。LC-DE 对有限可用残差的平方取平均，得到逐时预测能量。"],
], [100,WIDTH-100])
add("预测分布采用自由度 ν=5 的对角 Student-t，尺度 s 并非标准差。AdamW 优化观测位置的负对数似然（省略固定常数）：L = mean{log s + 3 log[1+(z-μ)²/(5s²)]}。用于监测的创新为 r=√(3/5)×(z-μ)/s；这里 μ、s 按步长对齐到当前目标时刻。", "small")
sub("3.3 同一运行内的初始化与监测")
add("以 0 为起点的行号表示：<b>[0,300)</b> 用于原始传感器的均值/标准差标准化；<b>[0,200)</b> 提供局部能量、方差和差分参照；<b>[200,300)</b> 提供五个分量的局部 median/MAD。<b>t≥300</b> 才产生可评估的监测分数。")
add("前 300 行会用整段初始化统计量回填标准化。因此，因果性陈述限定于初始化完成后的监测输出：分数仅依赖截至当前的可用观测。不能把热身期间或首批回算预测描述成在各历史预测原点已经真实发布的在线预测。", "small")

# Page 4: exact monitor math.
section("03","我们的模型","3.4 五类证据、稳健归一化与有限窗口扫描")
add("以下以已标准化观测 z(t) 表示原始序列，W(t) 为以 t 结束的 40 点窗口，R 为前 200 点参照。ε=10<super>-8</super>。TopKMean 表示对传感器维度取最大的 k 项后平均；冻结配置 k=1。")
table(["分量","实际计算与含义"],[
    ["c<sub>1</sub>：水平能量","|log[(mean<sub>W,j</sub> z²+ε)/(mean<sub>R,j</sub> z²+ε)]|；检测整体能量上升或下降。"],
    ["c<sub>2</sub>：局部均值","TopKMean<sub>j</sub>[(mean<sub>W</sub> z<sub>j</sub>)²]；检测相对标准化零点的局部均值偏移，并未再次减去前 200 点均值。"],
    ["c<sub>3</sub>：方差比","TopKMean<sub>j</sub>{log²[(var<sub>W</sub> z<sub>j</sub>+ε)/(var<sub>R</sub> z<sub>j</sub>+ε)]}；窗口方差使用样本方差。"],
    ["c<sub>4</sub>：差分能量比","TopKMean<sub>j</sub>{log²[(mean<sub>W</sub> (Δz<sub>j</sub>)²+ε)/(mean<sub>R</sub> (Δz<sub>j</sub>)²+ε)]}；反映局部变化速度与振荡强度。"],
    ["c<sub>5</sub>：预测能量比","先按预测步长和通道平均有限残差平方为 e(t)，再计算 |log[(mean<sub>W</sub> e+ε)/(mean<sub>R</sub> e+ε)]|。"],
], [107,WIDTH-107], size=8.3)
sub("局部稳健尺度与 top-2 融合")
add("对每个分量 j，在本次运行的 [200,300) 区间得到中位数 m<sub>j</sub> 和 MAD<sub>j</sub>；健康参照段的第 301 行起给出 g<sub>j</sub>=max(1.4826×MAD,0.05)。实际尺度及融合为：")
box("归一化与瞬时证据", "s<sub>j</sub> = max(1.4826 × MAD<sub>j</sub>, 0.25 × g<sub>j</sub>, 0.05)<br/>u<sub>t,j</sub> = clip[(c<sub>t,j</sub> - m<sub>j</sub>)/s<sub>j</sub>, 0, 20]<br/>p<sub>t</sub> = 五个 u<sub>t,j</sub> 中最大的两项的平均值")
sub("有限窗口累积与报警")
add("令 a<sub>t</sub>=p<sub>t</sub>-0.5。在最近最多 60 个已监测样本内，计算以当前时刻结束的各段累积和，取非负最大值：", "body")
add("S<sub>t</sub> = max{0, max<sub>1≤ℓ≤min(60,t-299)</sub> Σ<sub>i=t-ℓ+1</sub><super>t</super> a<sub>i</sub>}；　alarm<sub>t</sub> = I(S<sub>t</sub> &gt; h)", "formula")
add(f"完整模型冻结阈值 h={frozen['thresholds']['dual']:.6f}。阈值取健康校准段分数最大值；只在严格大于阈值时报警。相对于无限记忆的普通 CUSUM，这一实现只累积最近有限区间的证据，但有限记忆本身不构成跨工况误报保证。", "small")

# Page 5: dataset.
section("04","使用的数据集")
sub("4.1 主实验：SKAB v0.9.0")
add("本报告的 LC-DE 主结果来自 Skoltech Anomaly Benchmark 的 v0.9.0 固定版本。数据来自水循环实验平台的多变量传感器记录，包含 34 条带异常标签的实验文件和 1 条 anomaly-free 健康记录。[1] 本地核对共有 37,401 行带标签实验数据及 9,405 行健康参照数据，总计 46,806 行。")
table(["变量组","使用的传感器"],[
    ["振动","Accelerometer1RMS、Accelerometer2RMS"],
    ["电气","Current、Voltage"],
    ["压力与流量","Pressure、Volume Flow RateRMS（本地字段名）"],
    ["温度","Temperature、Thermocouple"],
], [105,WIDTH-105])
add("时间戳用于计算实际秒数；anomaly 标签用于评价异常区间。文件还提供 changepoint 标签，但本报告的有效检出以 anomaly=1 区间中的报警上升沿为准。窗口宽度以样本计，延迟以时间戳计，二者不应无条件互换。", "small")
sub("4.2 数据划分与用途")
table(["数据部分","数量 / 行范围（0 起点）","用途"],[
    ["健康模型拟合","4,702 行；[0,4702)","拟合 GRU；拟合全局原始数据基线"],
    ["健康分量参照","1,881 行；[4702,6583)","监测分量稳健尺度、全局残差参照"],
    ["健康阈值校准","1,411 行；[6583,7994)","固定各方法的报警阈值"],
    ["健康开发审计","1,411 行；[7994,9405)","查看健康段报警，不用于模型拟合"],
    ["异常开发实验","18 条：valve1 8 / valve2 2 / other 8","比较 24 个 LC-DE 候选配置"],
    ["异常测试实验","16 条：valve1 8 / valve2 2 / other 6","固定配置评价及后续统一基线比较"],
], [95,194,WIDTH-289],size=8)
sub("4.3 使用边界与数据问题")
add("开发/测试按完整实验文件划分，未将同一运行的相邻行随机分配到两侧。原始清单中的 valve1/16 和 valve2/4 分别映射到实际的 0 号文件；修正后的物理文件互不重叠。健康四段来自同一次实验，不能视作四个独立健康运行。")
add("测试文件 other/2.csv 在 0 起点第 104 行已出现异常，前 300 行含 196 个异常样本，违反健康初始化假设。因此主表保留全部 16 条，同时额外报告统一排除此文件后的 15 条敏感性分析。当前 LC-DE 尚无新的独立跨数据集确认结果；旧版本在 TEP、MetroPT 的记录不并入本报告成绩。", "small")

# Page 6: main baseline results.
section("05","我们的结果","5.1 固定方法、统一监测区间的完整基线比较（16 条）")
baseline_order=["global/ea_hx_cusum","global/predictive_residual_cusum","global/pca_t2_plus_q","global/energy_cusum",
                "self_starting/ea_hx_cusum","self_starting/predictive_residual_cusum","self_starting/pca_t2_plus_q","self_starting/energy_cusum","lcde_dual"]
names={"global/ea_hx_cusum":"EA-HX / 全局","global/predictive_residual_cusum":"预测残差 / 全局",
       "global/pca_t2_plus_q":"PCA T²+Q / 全局","global/energy_cusum":"能量 CUSUM / 全局",
       "self_starting/ea_hx_cusum":"EA-HX / 自校准","self_starting/predictive_residual_cusum":"预测残差 / 自校准",
       "self_starting/pca_t2_plus_q":"PCA T²+Q / 自校准","self_starting/energy_cusum":"能量 CUSUM / 自校准",
       "lcde_dual":"LC-DE / 完整模型","lcde_raw_only":"LC-DE / raw-only","lcde_predictive_only":"LC-DE / predictive-only"}
rows=[]
for k in baseline_order:
    r=A[k]
    rows.append([names[k],f"{r['mean_postwarmup_ap']:.4f}",f"{r['mean_postwarmup_f1']:.4f}",
                 f"{r['active_event_detected_runs']}/16",f"{r['active_event_median_delay_seconds']:g}",
                 f"{100*r['mean_healthy_sample_false_positive_rate']:.2f}%"])
table(["方法","AP ↑","F1 ↑","有效检出 ↑","延迟/s ↓","健康误报 ↓"], rows,
      [156,56,56,67,65,WIDTH-400], highlights=(8,),size=8)
add("注：有效检出指故障持续期间的报警上升沿；延迟仅在成功检出的运行内汇总。健康误报为逐实验健康样本报警比例的均值。EA-HX 是本项目旧方法对照，不能代表外部先进方法。", "small")
story.extend([ComparisonBars(),Spacer(1,4)])
sub("结果解读")
add("LC-DE 的 AP、F1 点估计高于这 8 个对照配置。相较 AP 最强的全局 PCA，平均 AP 增加 <b>0.0647</b>；相较 F1 最强的自校准预测残差，平均 F1 增加 <b>0.0804</b>。有效检出为 9/16，高于上述对照中的最高 8/16。[4]")
add("但优势不是所有指标同时成立：LC-DE 的健康样本误报率 6.75%，高于全局 PCA 的 1.23%；其有效检出延迟中位数为 87 秒，也并非最快。不同方法检出的实验子集不同，不能只凭条件延迟判定谁更优。", "small")
add("完整模型故障前误报为 0/16，但健康样本误报率仍非零：部分报警出现在故障恢复之后。这两项指标不可混用。", "small")

# Page 7: ablation and paired statistics.
section("05","我们的结果","5.2 消融：预测分支是否带来额外价值？")
rows=[]
for k in ("lcde_dual","lcde_raw_only","lcde_predictive_only"):
    r=A[k]
    rows.append([names[k].replace("LC-DE / ",""),f"{r['mean_postwarmup_ap']:.4f}",f"{r['mean_postwarmup_f1']:.4f}",
                 f"{r['active_event_detected_runs']}/16",f"{r['active_event_within60_runs']}/16",
                 f"{r['active_event_median_delay_seconds']:g}",f"{100*r['mean_healthy_sample_false_positive_rate']:.2f}%"])
table(["版本","AP ↑","F1 ↑","检出 ↑","60s内 ↑","延迟/s ↓","误报 ↓"],rows,
      [113,57,57,62,65,65,WIDTH-419],highlights=(0,),size=8)
add("raw-only 保留四个观测统计分量；predictive-only 仅保留预测能量比。三者使用同一冻结窗口配置，但分别以对应分数在健康校准段上的最大值设定阈值：完整模型 559.863460、raw-only 415.416608、predictive-only 431.252124。")
box("当前消融证据", "完整模型相比 raw-only，平均 F1 高 <b>0.0050</b>，健康样本误报率低约 <b>2.03 个百分点</b>，条件延迟中位数由 100 秒降至 87 秒；同时 AP 低 <b>0.0046</b>，少检出 <b>1 条</b>。这是性能取舍，尚不足以证明增加预测分支总体更优。")
sub("5.3 配对差值与不确定性")
add("下表为“LC-DE 完整模型减对照”的平均差值及运行级配对 bootstrap 95% 百分位区间，重采样 10,000 次。区间跨 0 表示该比较的差异方向尚不稳健；这些均为探索性结果，未做多重比较校正。")
pair_rows=[]
for label,key,metric in [("全局 PCA","global/pca_t2_plus_q","postwarmup_ap"),
                         ("自校准预测残差","self_starting/predictive_residual_cusum","postwarmup_ap"),
                         ("自校准预测残差","self_starting/predictive_residual_cusum","postwarmup_f1"),
                         ("raw-only","lcde_raw_only","postwarmup_ap"),
                         ("raw-only","lcde_raw_only","postwarmup_f1")]:
    v=result["paired_comparisons"][key][metric]
    lo,hi=v["paired_run_bootstrap_95_percentile_interval"]
    pair_rows.append([label,"AP" if metric.endswith("ap") else "F1",
                      f"{v['mean_difference_lcde_minus_comparator']:+.4f}",f"[{lo:+.4f}, {hi:+.4f}]"])
table(["对照","指标","平均差值","95% 区间"],pair_rows,[155,65,90,WIDTH-310])
add("对全局 PCA 的 AP 优势区间跨 0；对自校准预测残差的 F1 优势区间也跨 0。可以报告当前点估计和部分比较的积极信号，但不能写成“所有基线上的显著全面提升”。16 条实验仍来自同一平台，运行级重采样也不等于跨设备泛化证据。", "small")

# Page 8: complete run record and sensitivity.
section("05","我们的结果","5.4 完整模型的逐实验表现与敏感性分析")
rows=[]
for r in result["per_run"]["lcde_dual"]:
    delay="未检出" if r["active_event_delay_seconds"] is None else f"{r['active_event_delay_seconds']:g}"
    label=r["run_id"]+(" *" if r["commissioning_anomaly_rows"] else "")
    rows.append([label,"是" if r["active_event_detected"] else "否",delay,
                 f"{r['postwarmup_ap']:.3f}",f"{r['postwarmup_f1']:.3f}"])
table(["实验文件","故障期间检出","有效延迟/s","AP","F1"],rows,[167,95,95,73,WIDTH-430],size=8)
add("* other/2.csv 初始化已含异常。valve1/6.csv、valve1/1.csv 的旧口径首报分别在异常结束后 74 秒、156 秒出现，故修正后不计检出。完整模型由旧口径 11/16 改为 9/16；模型与阈值没有因此重调。", "small")
sub("排除初始化异常的 15 条敏感性结果")
rows=[]
for key in ("lcde_dual","lcde_raw_only","global/pca_t2_plus_q","self_starting/predictive_residual_cusum"):
    r=C[key]
    rows.append([names[key],f"{r['mean_postwarmup_ap']:.4f}",f"{r['mean_postwarmup_f1']:.4f}",f"{r['active_event_detected_runs']}/15"])
table(["方法","AP","F1","有效检出"],rows,[WIDTH-210,70,70,70],highlights=(0,),size=8)
add("所有方法采用同一排除规则，并与全部 16 条主结果并列报告。结论方向一致：LC-DE 相对传统基线具有点估计优势，但 raw-only 的 AP 与检出数仍更高。", "small")

# Page 9: conclusion, interpretation, reproducibility.
section("05","我们的结果","5.5 阶段结论、证据边界与复现来源")
sub("已经得到的结果")
add("当前实现提供了完整的“局部参照统计量、健康多步预测残差、稳健尺度、top-2 融合、有限窗口扫描”监测链。GPU 完成预测模型训练及残差推理；统计特征与扫描由 NumPy 在 CPU 上计算。不能将整个流水线描述成每一步都在 GPU 上执行。")
add("在本项目统一条件下，LC-DE 完整模型在 AP、F1 和有效检出数的点估计上超过已运行的传统基线。对应比较前的 144 条开发路径复现误差为 0；指标修正相关回归测试记录为 40 项通过。测试通过支持实现一致性，不是不存在任何潜在代码问题的证明。")
sub("尚未得到的结论")
add("创新性目前应表述为针对本任务的模块组合与监测设计，尚未完成足够的文献查新与更强模型对比，不能宣称首创或 SOTA。预测分支的净收益仍待证明；raw-only 在 AP 与检出数上略占优，完整模型在误报、F1 与条件延迟上有所改善。")
add("本次测试在配置固定后运行，但启用测试的比较规则经过开发结果后调整；后续还统一修正了评价口径。因此应称为研究阶段的固定模型测试与回顾性统一比较，不能改称原始预注册方案全部通过的确认实验。已查看的测试标签不能继续承担调参后的独立验证。")
box("建议的下一阶段", "先在开发数据上检验局部对照、top-2 融合和有限窗口扫描各自的必要性，并补齐更强的异常检测基线。在新的独立划分或数据上验证候选方案，重点回答“预测分支是否值得保留”。原先的工程门槛留作参考，当前以公平对比与模块增益为主。")
sub("资料与结果来源")
refs=[
    "[1] Katser, I. D.; Kozitsin, V. O. Skoltech Anomaly Benchmark (SKAB), 2020. DOI: 10.34740/KAGGLE/DSV/1693952。固定版本 v0.9.0；官方仓库：<link href='https://github.com/waico/SKAB' color='#007E87'>https://github.com/waico/SKAB</link>。",
    "[2] 模型实现：src/mate_spc_cris_v9/local_contrast.py；预测骨干：src/mate_spc_cris_v9/model.py。",
    "[3] 冻结方法：outputs/lcde_skab_v2_self_calibrated/frozen_method.json；预测权重：outputs/ea_hx_skab/ea_hx_skab_model.npz。",
    "[4] 本报告数值主来源：outputs/skab_frozen_baseline_comparison/comparison_result.json（2026-10-13 UTC）；比较脚本：scripts/compare_skab_frozen_baselines_gpu.py。",
    "[5] 原始测试记录：outputs/lcde_skab_v2_self_calibrated/final_result.json；其旧评价口径以本报告 [4] 中统一修正后的字段为准。",
]
for ref in refs:
    add(ref,"small")
add("项目根目录：" + escape(ROOT.as_posix()), "small")

doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=46,leftMargin=46,
    topMargin=60,bottomMargin=53, title="LC-DE 模型与实验结果报告",
    author="MATH 项目研究记录", pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(PDF))
texts=[page.extract_text() or "" for page in reader.pages]
if len(reader.pages)!=9:
    raise RuntimeError(f"Expected 9 deliberately laid-out pages, got {len(reader.pages)}")
for needle in ("背景","介绍","我们的模型","使用的数据集","我们的结果","0.8266","9/16"):
    if needle not in "\n".join(texts):
        raise RuntimeError(f"Missing report content: {needle}")
metadata={"pdf":str(PDF),"page_count":len(reader.pages),
          "result_source_sha256":hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest(),
          "pdf_sha256":hashlib.sha256(PDF.read_bytes()).hexdigest(),
          "page_text_lengths":[len(t) for t in texts]}
(OUT / "lcde_report_manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metadata,ensure_ascii=False,indent=2))
