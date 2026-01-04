# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle

# ========== 基本配置 ==========
EXCEL_FILE = "2015-2025咖啡进口数据及季节性图.xlsx"   # 改成你的文件路径
SHEET_NAME = "进口量分省"                           # 右侧为“年月×省份”的那张表
OUT_DIR     = "seasonal_plots_province"            # 导出目录
PDF_NAME    = "咖啡_分省份_季节性图.pdf"            # 合并导出 PDF 文件名

# 只画这些省份（留空则画表里全部省份列）
ONLY_PROVINCES = []   # 例: ["北京市", "上海市", "江苏省", "山东省", "福建省", "广东省", "安徽省"]

# 这些列名视为“总计/合计”，默认不画
TOTAL_COL_CANDIDATES = {"总计", "合计", "全国", "总量", "统计", "总计1", "总数"}

# ========== 中文字体（自动选择系统可用的一个） ==========
def set_chinese_font():
    candidates = [
        "Songti SC", "PingFang HK", "STHeiti", "Heiti TC", "Hei",
        "Kaiti SC", "HanziPen SC", "Lantinghei SC", "Weibei SC",
    ]
    avail = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in avail:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False

set_chinese_font()

# ========== 小工具 ==========
def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))

def to_numeric_series(s: pd.Series) -> pd.Series:
    """把可能包含千分位逗号/全角空格/破折号的列转为浮点"""
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("\u3000", "", regex=False)  # 全角空格
         .str.replace("—", "", regex=False)       # 破折号
         .str.strip(),
        errors="coerce"
    )

def clean_yearmonth_column(sr: pd.Series) -> pd.Series:
    """
    清洗“日期/年月”列：
    - 只保留形如 YYYYMM 的 6 位数字
    - 其余（如“2015年度”“全国占比”等标签）全部丢弃
    """
    s = sr.astype(str).str.strip()
    # 先尝试直接匹配 6 位数字；否则从文本中提取 6 位数字
    six = s.where(s.str.fullmatch(r"\d{6}")).fillna(
        s.str.extract(r"(\d{6})", expand=False)
    )
    return six

# ========== 读取与列识别（分省份） ==========
df0 = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=0)
df0.columns = [str(c).strip() for c in df0.columns]

# 找到“日期/年月”列（不假设在第 1 列）
date_candidates = [c for c in df0.columns if ("日期" in c) or ("年月" in c)]
if not date_candidates:
    raise ValueError("未找到‘日期/年月’列，请检查表头。")
date_col = date_candidates[0]
date_idx = df0.columns.get_loc(date_col)

# —— 剔除“2015年度/全国占比”等标签行：只保留能清洗出 6 位 YYYYMM 的行 ——
df0["_YM_"] = clean_yearmonth_column(df0[date_col])
df0 = df0[~df0["_YM_"].isna()].copy()

# 从“日期/年月”右侧全部视为省份列（与你截图结构一致）
province_cols = list(df0.columns[date_idx + 1 :])
# 过滤掉“总计/合计/全国”等列
province_cols = [c for c in province_cols if c not in TOTAL_COL_CANDIDATES]
# 只画用户指定的省份（如配置了）
if ONLY_PROVINCES:
    province_cols = [c for c in province_cols if c in ONLY_PROVINCES]
if not province_cols:
    raise ValueError("未检测到可绘制的省份列，请检查表头或 ONLY_PROVINCES 配置。")

# ========== 解析 年份 / 月份 ==========
# 使用清洗后的 _YM_ 列，避免受标签行影响
df0["日期"]  = pd.to_datetime(df0["_YM_"], format="%Y%m", errors="coerce")
df0["年份"]  = df0["日期"].dt.year
df0["月份"]  = df0["日期"].dt.month
df = df0.dropna(subset=["年份", "月份"]).copy()
df["年份"] = df["年份"].astype(int)
df["月份"] = df["月份"].astype(int)

# ========== 清洗为纯数值（去逗号/全角空格/破折号） ==========
for c in province_cols:
    df[c] = to_numeric_series(df[c])

# ========== 分省份：绘图（配色与“分国别”一致：tab20 + cycle） ==========
os.makedirs(OUT_DIR, exist_ok=True)

def plot_one_province(province: str, pdf: PdfPages | None = None):
    sdf = df[["年份", "月份", province]].dropna(subset=["月份"]).copy()
    sdf = sdf.sort_values(["年份", "月份"])

    plt.figure(figsize=(10, 6))

    color_list = plt.cm.tab20.colors   # 20 种固定颜色
    color_cycler = cycle(color_list)   # 按年份顺序逐个取色（超过 20 年会循环）

    for year, g in sdf.groupby("年份", sort=True):
        color = next(color_cycler)
        full = pd.DataFrame({"月份": range(1, 13)})
        g2 = full.merge(g[["月份", province]], on="月份", how="left")
        plt.plot(
            g2["月份"],
            g2[province],
            marker="o",
            label=str(year),
            color=color
        )

    plt.title(f"{province} 进口量季节性图")
    plt.xlabel("月份 (1-12)")
    plt.ylabel("进口量 (吨)")
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    plt.legend(title="年份", ncol=3, fontsize=9)
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, f"{safe_name(province)}_进口量季节性图.png")
    plt.savefig(png_path, dpi=300)
    if pdf is not None:
        pdf.savefig()
    plt.close()

# ========== 批量输出 PNG + 多页 PDF（分省份）==========
with PdfPages(os.path.join(OUT_DIR, PDF_NAME)) as pdf:
    for prov in province_cols:
        plot_one_province(prov, pdf=pdf)

print(f"✅ 分省份：已生成 {len(province_cols)} 张 PNG 于: {OUT_DIR}")
print(f"✅ 分省份：并导出 PDF: {os.path.join(OUT_DIR, PDF_NAME)}")

# ========== 出口量分国别：季节性图（与“分国别”配色一致） ==========
def seasonal_export_by_country(
    excel_file: str = EXCEL_FILE,
    sheet_name: str = "出口量分国别",
    out_dir: str = "seasonal_plots_export_country",
    pdf_name: str = "咖啡_出口量_分国别_季节性图.pdf",
    only_countries: list[str] | None = None,
    drop_total_like: set[str] = {"总计","合计","全国","总量","统计"}
):
    set_chinese_font()
    os.makedirs(out_dir, exist_ok=True)

    dfc = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
    dfc.columns = [str(c).strip() for c in dfc.columns]

    # 找“日期/年月”列（不假设在第1列）
    date_candidates = [c for c in dfc.columns if ("日期" in c) or ("年月" in c)]
    if not date_candidates:
        # 兜底：默认第一列就是年月
        date_candidates = [dfc.columns[0]]
    date_col = date_candidates[0]
    date_idx = dfc.columns.get_loc(date_col)

    # 剔除“201x年度 / 全国占比”等标签行
    dfc["_YM_"] = clean_yearmonth_column(dfc[date_col])
    dfc = dfc[~dfc["_YM_"].isna()].copy()

    # 从日期右侧 ~ 末尾 视为国家列，并过滤总计/合计
    country_cols = [c for c in dfc.columns[date_idx + 1:] if c not in drop_total_like]
    if only_countries:
        country_cols = [c for c in country_cols if c in only_countries]
    if not country_cols:
        raise ValueError("未检测到可绘制的国家列，请检查表头或 only_countries 参数。")

    # 解析年月 -> 年份/月份
    dfc["日期"] = pd.to_datetime(dfc["_YM_"], format="%Y%m", errors="coerce")
    dfc["年份"] = dfc["日期"].dt.year
    dfc["月份"] = dfc["日期"].dt.month
    dfc = dfc.dropna(subset=["年份","月份"]).copy()
    dfc["年份"] = dfc["年份"].astype(int)
    dfc["月份"] = dfc["月份"].astype(int)

    # 数值化各国列
    for c in country_cols:
        dfc[c] = to_numeric_series(dfc[c])

    # —— 与分国别脚本一致的配色：tab20 + cycle（每张图内按年份顺序取色）——
    def plot_one_country(country: str, pdf: PdfPages | None = None, fill_na: bool = False):
        sdf = dfc[["年份", "月份", country]].dropna(subset=["月份"]).sort_values(["年份", "月份"])
        plt.figure(figsize=(10, 6))

        color_list = plt.cm.tab20.colors
        color_cycler = cycle(color_list)

        for year, g in sdf.groupby("年份", sort=True):
            color = next(color_cycler)
            full = pd.DataFrame({"月份": range(1, 13)})
            g2 = full.merge(g[["月份", country]], on="月份", how="left")

            y = g2[country]
            if fill_na:
                # 如需不留空可选用插值/前向填充/0，保持默认 False = 缺失断线
                y = y.interpolate(limit_direction="both")

            plt.plot(
                g2["月份"],
                y,
                marker="o",
                label=str(year),
                color=color
            )

        plt.title(f"{country} 出口量季节性图")
        plt.xlabel("月份 (1-12)")
        plt.ylabel("出口量 (吨)")
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.legend(title="年份", ncol=3, fontsize=9)
        plt.tight_layout()

        png_path = os.path.join(out_dir, f"{safe_name(country)}_出口量季节性图.png")
        plt.savefig(png_path, dpi=300)
        if pdf is not None:
            pdf.savefig()
        plt.close()

    # 批量导出
    with PdfPages(os.path.join(out_dir, pdf_name)) as pdf:
        for country in country_cols:
            plot_one_country(country, pdf=pdf)

    print(f"✅ 出口量分国别：已生成 {len(country_cols)} 张 PNG 到 {out_dir}")
    print(f"✅ 出口量分国别：并导出多页 PDF：{os.path.join(out_dir, pdf_name)}")

# ==== 调用（直接运行脚本即生成两套图）====
seasonal_export_by_country(
    excel_file=EXCEL_FILE,
    sheet_name="主要省份至主要国家出口量 ",       # 如果你的 sheet 名不同，改这里
    out_dir="seasonal_plots_export_country",
    pdf_name="咖啡_出口量_分国别_季节性图.pdf",
    only_countries=[]               # 例如 ["美国","越南","德国"]；留空画全部
)
# ========= 出口量·分省份：季节性图 =========
def seasonal_export_by_province(
    excel_file: str = EXCEL_FILE,
    sheet_name: str = "出口量分省分",          # 若你的表名是“出口量分省”，改这里
    out_dir: str = "seasonal_plots_export_province",
    pdf_name: str = "咖啡_出口量_分省份_季节性图.pdf",
    only_provinces: list[str] | None = None,
    drop_total_like: set[str] = {"总计","合计","全国","总量","统计"},
    fill_method: str = "ffill"  # "none" 断线；"zero"/"ffill"/"interpolate" 连线
):
    set_chinese_font()
    os.makedirs(out_dir, exist_ok=True)

    # 读取
    dfp = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
    dfp.columns = [str(c).strip() for c in dfp.columns]

    # 找“日期/年月”列
    date_candidates = [c for c in dfp.columns if ("日期" in c) or ("年月" in c)]
    if not date_candidates:
        # 兜底：默认第一列
        date_candidates = [dfp.columns[0]]
    date_col = date_candidates[0]
    date_idx = dfp.columns.get_loc(date_col)

    # 剔除 “201X年度 / 全国占比” 等行标签：只保留能抽出6位YYYYMM的
    def clean_yearmonth_column(sr: pd.Series) -> pd.Series:
        s = sr.astype(str).str.strip()
        six = s.where(s.str.fullmatch(r"\d{6}")).fillna(s.str.extract(r"(\d{6})", expand=False))
        return six
    dfp["_YM_"] = clean_yearmonth_column(dfp[date_col])
    dfp = dfp[~dfp["_YM_"].isna()].copy()

    # 从日期右侧到末尾视为省份列，并过滤“总计/合计/全国”
    province_cols = [c for c in dfp.columns[date_idx+1:] if c not in drop_total_like]
    if only_provinces:
        province_cols = [c for c in province_cols if c in only_provinces]
    if not province_cols:
        raise ValueError("未检测到可绘制的省份列，请检查表头或 only_provinces。")

    # 解析年月 → 年/月
    dfp["日期"] = pd.to_datetime(dfp["_YM_"], format="%Y%m", errors="coerce")
    dfp["年份"] = dfp["日期"].dt.year
    dfp["月份"] = dfp["日期"].dt.month
    dfp = dfp.dropna(subset=["年份","月份"]).copy()
    dfp["年份"] = dfp["年份"].astype(int)
    dfp["月份"] = dfp["月份"].astype(int)

    # 数值化
    def to_numeric_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False)
                          .str.replace("\u3000", "", regex=False)
                          .str.replace("—", "", regex=False)
                          .str.strip(),
            errors="coerce"
        )
    for c in province_cols:
        dfp[c] = to_numeric_series(dfp[c])

    # 缺失值填充
    def fill_series(y: pd.Series) -> pd.Series:
        if fill_method == "zero":
            return y.fillna(0)
        if fill_method == "ffill":
            return y.ffill().bfill()
        if fill_method == "interpolate":
            return y.interpolate(limit_direction="both")
        return y  # "none"：保持断线

    # 配色：与分国别一致
    def plot_one_province(province: str, pdf: PdfPages | None = None):
        sdf = dfp[["年份","月份",province]].dropna(subset=["月份"]).sort_values(["年份","月份"])
        plt.figure(figsize=(10, 6))
        color_list = plt.cm.tab20.colors
        color_cycler = cycle(color_list)

        for year, g in sdf.groupby("年份", sort=True):
            color = next(color_cycler)
            full = pd.DataFrame({"月份": range(1, 13)})
            g2 = full.merge(g[["月份", province]], on="月份", how="left")
            y = fill_series(g2[province])

            plt.plot(g2["月份"], y, marker="o", label=str(year), color=color)

        plt.title(f"{province} 出口量季节性图")
        plt.xlabel("月份 (1-12)")
        plt.ylabel("出口量 (吨)")
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.legend(title="年份", ncol=3, fontsize=9)
        plt.tight_layout()

        png_path = os.path.join(out_dir, f"{re.sub(r'[\\/:*?\"<>|]+','_',province)}_出口量季节性图.png")
        plt.savefig(png_path, dpi=300)
        if pdf is not None:
            pdf.savefig()
        plt.close()

    # 导出
    with PdfPages(os.path.join(out_dir, pdf_name)) as pdf:
        for prov in province_cols:
            plot_one_province(prov, pdf=pdf)

    print(f"✅ 出口量·分省份：已生成 {len(province_cols)} 张 PNG 到 {out_dir}")
    print(f"✅ 出口量·分省份：并导出多页 PDF：{os.path.join(out_dir, pdf_name)}")
seasonal_export_by_province(
    excel_file=EXCEL_FILE,
    sheet_name="主要省份出口量占比",   # 如果你的工作表叫“出口量分省”，就在这里改成那个名字
    out_dir="seasonal_plots_export_province",
    pdf_name="咖啡_出口量_分省份_季节性图.pdf",
    only_provinces=[],            # 例如 ["上海市","江苏省","广东省"]，留空画全部
    fill_method="zero"            # "none"/"zero"/"ffill"/"interpolate"
)
