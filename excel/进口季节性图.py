# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
from itertools import cycle

# ========= 配置 =========
EXCEL_FILE = "2015-2025咖啡进口数据及季节性图（启用宏）.xlsm"  # 改成你的文件路径
SHEET_NAME = "进口量分国别"                              # 表名
OUT_DIR = "seasonal_plots"                              # 输出目录
PDF_NAME = "咖啡_各国_季节性图_2015-2025.pdf"             # 合并导出 PDF
ONLY_COUNTRIES = []                                     # 指定国家列表；为空则画全部

# ========= 中文字体（自动选择系统可用的一个） =========
def set_chinese_font():
    candidates = [
        "Songti SC", "PingFang HK", "STHeiti", "Heiti TC", "Hei",
        "Kaiti SC", "HanziPen SC", "Lantinghei SC", "Weibei SC",
        "Yuanti SC", "Wawati SC"
    ]
    avail = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in avail:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False

set_chinese_font()

# ========= 读取与清洗 =========
df0 = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=0)
df0.columns = [str(c).strip() for c in df0.columns]

# ✅ 如果有重复列名，先去掉右边的重复列
if df0.columns.duplicated().any():
    df0 = df0.loc[:, ~df0.columns.duplicated()]

# 找“日期/年月”列
date_candidates = [c for c in df0.columns if ("日期" in c) or ("年月" in c)]
date_col = date_candidates[0] if date_candidates else df0.columns[0]
date_idx = df0.columns.get_loc(date_col)

# ✅ 删除已有的「年份」「月份」「日期」列（避免与我们新生成的冲突）
for col in ("年份", "月份", "日期"):
    if col in df0.columns and col != date_col:
        df0 = df0.drop(columns=[col])

# 抽取 6 位年月
def clean_yearmonth_column(sr: pd.Series) -> pd.Series:
    s = sr.astype(str).str.strip()
    return s.where(s.str.fullmatch(r"\d{6}")).fillna(
        s.str.extract(r"(\d{6})", expand=False)
    )

df0["_YM_"] = clean_yearmonth_column(df0[date_col])
df0 = df0[~df0["_YM_"].isna()].copy()

# 解析年月
df0["日期"] = pd.to_datetime(df0["_YM_"], format="%Y%m", errors="coerce")
df0["年份"] = df0["日期"].dt.year
df0["月份"] = df0["日期"].dt.month
df = df0.dropna(subset=["年份", "月份"]).copy()
df["年份"] = df["年份"].astype(int)
df["月份"] = df["月份"].astype(int)

# 数值化各国列
value_cols = [c for c in df.columns if c not in ["年份", "月份", "日期", "_YM_"]]
for c in value_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 只画指定国家
if ONLY_COUNTRIES:
    draw_countries = [c for c in value_cols if c in ONLY_COUNTRIES]
else:
    draw_countries = value_cols

# 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 画图 =========
def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name)

def plot_one_country(country: str, pdf: PdfPages | None = None):
    sdf = df[["年份", "月份", country]].dropna(subset=["月份"]).copy()
    sdf = sdf.sort_values(["年份", "月份"])

    plt.figure(figsize=(10, 6))

    color_list = plt.cm.tab20.colors
    color_cycler = cycle(color_list)

    for year, g in sdf.groupby("年份", sort=True):
        color = next(color_cycler)
        full = pd.DataFrame({"月份": range(1, 13)})
        g = full.merge(g[["月份", country]], on="月份", how="left")

        # 缺失值保持 NaN，不连线（比如 2025 年 5–7 月）
        plt.plot(
            g["月份"], g[country],
            marker="o", label=str(year), color=color
        )

    plt.title(f"{country}进口量季节性图")
    plt.xlabel("月份 (1至12)")
    plt.ylabel("进口量 (吨)")
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    plt.legend(title="年份", ncol=3, fontsize=9)
    plt.tight_layout()

    # PNG 单独导出
    png_path = os.path.join(OUT_DIR, f"{safe_name(country)}_进口量季节性图.png")
    plt.savefig(png_path, dpi=300)

    if pdf is not None:
        pdf.savefig()
    plt.close()

# ========= 批量输出 =========
with PdfPages(os.path.join(OUT_DIR, PDF_NAME)) as pdf:
    for country in draw_countries:
        plot_one_country(country, pdf=pdf)

print(f"✅ 已生成 {len(draw_countries)} 张 PNG 到文件夹：{OUT_DIR}")
print(f"✅ 已导出多页 PDF：{os.path.join(OUT_DIR, PDF_NAME)}")
