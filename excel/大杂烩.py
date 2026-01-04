# -*- coding: utf-8 -*-
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle

# ===================== 运行开关 & 基本配置 =====================
EXCEL_FILE = "2015-2025咖啡进口数据及季节性图（启用宏）.xlsm"   # ← 改成你的文件路径（支持 .xlsx/.xlsm）
RUN_IMPORT_BY_COUNTRY  = True   # 分国别·进口
RUN_IMPORT_BY_PROVINCE = True   # 分省份·进口
RUN_EXPORT_BY_COUNTRY  = True   # 分国别·出口
RUN_EXPORT_BY_PROVINCE = True  # 分省份·出 口

# 各 sheet 名（按你的工作簿命名调整）
SHEET_IMPORT_COUNTRY   = "进口量分国别"
SHEET_IMPORT_PROVINCE  = "进口量分省"      # 或 “进口量分省分”
SHEET_EXPORT_COUNTRY   = "主要出口国家"
SHEET_EXPORT_PROVINCE  = "主要省份出口量占比"    # 或 “主要省份出口量占比”

# 输出目录
OUT_DIR_BASE = "seasonal_plots"  # 各类图会在此目录下分子目录输出

# 缺月填充策略： "none"(断线) / "zero"(补0) / "ffill" / "interpolate"
FILL_METHOD_IMPORT_COUNTRY  = "none"
FILL_METHOD_IMPORT_PROVINCE = "none"
FILL_METHOD_EXPORT_COUNTRY  = "none"
FILL_METHOD_EXPORT_PROVINCE = "zero"

# 可只画部分对象（留空画全部）
ONLY_COUNTRIES: list[str] = []  # 例：["印度尼西亚", "越南", "巴西"]
ONLY_PROVINCES: list[str] = []  # 例：["北京市","上海市","江苏省","山东省","福建省","广东省","安徽省"]

# 识别为“汇总”的列名（会被剔除）
DROP_TOTAL_LIKE = {"总计","合计","全国","总量","统计","总计1","总数"}

# ===================== 字体 & 工具 =====================
def set_chinese_font():
    candidates = ["Songti SC","PingFang HK","STHeiti","Heiti TC","Hei",
                  "Kaiti SC","HanziPen SC","Lantinghei SC","Weibei SC","Yuanti SC","Wawati SC"]
    avail = {f.name for f in fm.fontManager.ttflist}
    for f in candidates:
        if f in avail:
            plt.rcParams["font.sans-serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False

set_chinese_font()

def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))

def clean_yearmonth_column(sr: pd.Series) -> pd.Series:
    """只保留 6 位 YYYYMM；把 '2015年度'、'全国占比' 等标签行过滤掉。"""
    s = sr.astype(str).str.strip()
    return s.where(s.str.fullmatch(r"\d{6}")).fillna(s.str.extract(r"(\d{6})", expand=False))

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("\u3000", "", regex=False)
         .str.replace("—", "", regex=False)
         .str.strip(),
        errors="coerce"
    )

def fill_series(y: pd.Series, method: str) -> pd.Series:
    if method == "zero":
        return y.fillna(0)
    if method == "ffill":
        return y.ffill().bfill()
    if method == "interpolate":
        return y.interpolate(limit_direction="both")
    return y  # "none": 保持 NaN → 断线

# ===================== 读取宽表的通用函数 =====================
def load_wide_table(
    excel_file: str, sheet_name: str,
    id_keywords=("日期","年月"), drop_total_like: set[str]=DROP_TOTAL_LIKE
):
    """
    读取“YYYYMM + 多列数值”的交叉表，并返回：
    df: 含 年份/月份/日期 及数值列
    value_cols: 需要绘图的列（自动剔除汇总列）
    date_idx: 日期列索引（用于定位日期右侧的列范围）
    """
    df0 = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)  # 支持 xlsm/xlsx
    df0.columns = [str(c).strip() for c in df0.columns]

    # 去掉重复列名（保留最左边）
    if df0.columns.duplicated().any():
        df0 = df0.loc[:, ~df0.columns.duplicated()]

    # 找日期/年月列（不固定位置/名字）；若没有则默认第一列
    date_candidates = [c for c in df0.columns if any(k in c for k in id_keywords)]
    date_col = date_candidates[0] if date_candidates else df0.columns[0]
    date_idx = df0.columns.get_loc(date_col)

    # 为避免与我们生成的“年份/月份/日期”冲突，把同名列删掉（保留原始 date_col）
    for col in ("年份","月份","日期"):
        if col in df0.columns and col != date_col:
            df0 = df0.drop(columns=[col])

    # 过滤标签行，抽取 YYYYMM
    df0["_YM_"] = clean_yearmonth_column(df0[date_col])
    df0 = df0[~df0["_YM_"].isna()].copy()

    # 解析年月
    df0["日期"] = pd.to_datetime(df0["_YM_"], format="%Y%m", errors="coerce")
    df0["年份"] = df0["日期"].dt.year
    df0["月份"] = df0["日期"].dt.month
    df = df0.dropna(subset=["年份","月份"]).copy()
    df["年份"] = df["年份"].astype(int)
    df["月份"] = df["月份"].astype(int)

    # 数值列 = 日期列右侧到末尾，剔除汇总列 + 我们的辅助列
    candidates = [c for c in df.columns[date_idx+1:] if c not in drop_total_like]
    value_cols = [c for c in candidates if c not in {"_YM_","日期","年份","月份"}]

    # 数值化
    for c in value_cols:
        df[c] = to_numeric_series(df[c])

    return df, value_cols, date_idx

# ===================== 通用绘图器（每列一张图） =====================
def plot_entity_series(
    df: pd.DataFrame,
    value_cols: list[str],
    out_dir: str,
    pdf_name: str,
    title_tpl: str,              # 如 "{entity} 进口量季节性图"
    ylabel: str = "进口量 (吨)",
    only_list: list[str] | None = None,
    fill_method: str = "none"
):
    os.makedirs(out_dir, exist_ok=True)

    # 是否筛选
    cols = value_cols
    if only_list:
        only = set(only_list)
        cols = [c for c in value_cols if c in only]
    if not cols:
        raise ValueError("没有可绘制的列（筛选后为空）。")

    def plot_one(entity: str, pdf: PdfPages | None = None):
        sdf = df[["年份","月份",entity]].dropna(subset=["月份"]).sort_values(["年份","月份"])

        plt.figure(figsize=(10, 6))
        color_list = plt.cm.tab20.colors
        color_cycler = cycle(color_list)

        for year, g in sdf.groupby("年份", sort=True):
            color = next(color_cycler)
            full = pd.DataFrame({"月份": range(1, 13)})
            g2 = full.merge(g[["月份", entity]], on="月份", how="left")
            y = fill_series(g2[entity], fill_method)
            plt.plot(g2["月份"], y, marker="o", label=str(year), color=color)

        plt.title(title_tpl.format(entity=entity))
        plt.xlabel("月份 (1-12)")
        plt.ylabel(ylabel)
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.legend(title="年份", ncol=3, fontsize=9)
        plt.tight_layout()

        png_path = os.path.join(out_dir, f"{safe_name(entity)}.png")
        plt.savefig(png_path, dpi=300)
        if pdf:
            pdf.savefig()
        plt.close()

    with PdfPages(os.path.join(out_dir, pdf_name)) as pdf:
        for ent in cols:
            plot_one(ent, pdf=pdf)

    print(f"✅ 输出 {len(cols)} 张 PNG 到：{out_dir}")
    print(f"✅ 合并 PDF：{os.path.join(out_dir, pdf_name)}")

# ===================== 1) 分国别·进口 =====================
if RUN_IMPORT_BY_COUNTRY:
    df_ic, cols_ic, _ = load_wide_table(EXCEL_FILE, SHEET_IMPORT_COUNTRY)
    out_dir = os.path.join(OUT_DIR_BASE, "import_by_country")
    plot_entity_series(
        df_ic, cols_ic, out_dir,
        pdf_name="咖啡_进口量_分国别_季节性图.pdf",
        title_tpl="{entity} 进口量季节性图",
        ylabel="进口量 (吨)",
        only_list=ONLY_COUNTRIES,
        fill_method=FILL_METHOD_IMPORT_COUNTRY
    )

# ===================== 2) 分省份·进口 =====================
if RUN_IMPORT_BY_PROVINCE:
    df_ip, cols_ip, _ = load_wide_table(EXCEL_FILE, SHEET_IMPORT_PROVINCE)
    out_dir = os.path.join(OUT_DIR_BASE, "import_by_province")
    plot_entity_series(
        df_ip, cols_ip, out_dir,
        pdf_name="咖啡_进口量_分省份_季节性图.pdf",
        title_tpl="{entity} 进口量季节性图",
        ylabel="进口量 (吨)",
        only_list=ONLY_PROVINCES,
        fill_method=FILL_METHOD_IMPORT_PROVINCE
    )

# ===================== 3) 分国别·出口 =====================
if RUN_EXPORT_BY_COUNTRY:
    df_ec, cols_ec, _ = load_wide_table(EXCEL_FILE, SHEET_EXPORT_COUNTRY)
    out_dir = os.path.join(OUT_DIR_BASE, "export_by_country")
    plot_entity_series(
        df_ec, cols_ec, out_dir,
        pdf_name="咖啡_出口量_分国别_季节性图.pdf",
        title_tpl="{entity} 出口量季节性图",
        ylabel="出口量 (吨)",
        only_list=ONLY_COUNTRIES,
        fill_method=FILL_METHOD_EXPORT_COUNTRY
    )

# ===================== 4) 分省份·出口（可选） =====================
if RUN_EXPORT_BY_PROVINCE:
    df_ep, cols_ep, _ = load_wide_table(EXCEL_FILE, SHEET_EXPORT_PROVINCE)
    out_dir = os.path.join(OUT_DIR_BASE, "export_by_province")
    plot_entity_series(
        df_ep, cols_ep, out_dir,
        pdf_name="咖啡_出口量_分省份_季节性图.pdf",
        title_tpl="{entity} 出口量季节性图",
        ylabel="出口量 (吨)",
        only_list=ONLY_PROVINCES,
        fill_method=FILL_METHOD_EXPORT_PROVINCE
    )
