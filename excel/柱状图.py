import pandas as pd
from pathlib import Path

# 1. 读取数据
src = Path("FAOSTAT_data_en_8-28-2025.xlsx")
df = pd.read_excel(src, sheet_name=0)

# 2. 统一列名
rename_map = {}
for c in df.columns:
    lc = str(c).strip().lower()
    if lc == "area":
        rename_map[c] = "Area"
    elif lc == "element":
        rename_map[c] = "Element"
    elif lc == "element code":
        rename_map[c] = "Element Code"
    elif lc == "item":
        rename_map[c] = "Item"
    elif lc == "year":
        rename_map[c] = "Year"
    elif lc == "unit":
        rename_map[c] = "Unit"
    elif lc == "value":
        rename_map[c] = "Value"
df = df.rename(columns=rename_map)

# 3. 仅保留 Coffee, green 且 Element 是 Yield 或 Area harvested
df1 = df[df["Item"].astype(str).str.strip().str.lower() == "coffee, green"]
df1 = df1[df1["Element"].isin(["Yield", "Area harvested"])]

# 4. 透视成长表
pvt = df1.pivot_table(index="Year", columns=["Area", "Element"], values="Value", aggfunc="first")

# 5. 指定国家顺序
countries = ["Brazil", "Indonesia", "Uganda", "Ethiopia", "Viet Nam", "Colombia"]

# 6. 按国家生成列顺序
ordered_cols = []
for c in countries:
    for e in ["Yield", "Area harvested"]:
        if (c, e) in pvt.columns:
            ordered_cols.append((c, e))
rest = [c for c in pvt.columns if c not in ordered_cols]
rest = sorted(rest, key=lambda x: (x[0], x[1]))
final_cols = ordered_cols + rest
pvt = pvt[final_cols]


# 7. 格式化表头
def make_header(area, elem):
    sample = df1[(df1["Area"] == area) & (df1["Element"] == elem)].head(1)
    code = str(sample["Element Code"].iloc[0]) if len(sample) > 0 else ""
    unit = str(sample["Unit"].iloc[0]) if len(sample) > 0 else ""
    item = str(sample["Item"].iloc[0]) if len(sample) > 0 else "Coffee, green"
    return f"{elem} ({code}) — {area} — {item} ({unit})"


pvt.columns = [make_header(a, e) for a, e in pvt.columns]
pvt = pvt.reset_index().rename(columns={"Year": "period"}).sort_values("period")

# 8. 导出到 Excel（美化）
out_path = Path("咖啡作物_FAOSTAT_整理_截图格式.xlsx")
with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    sheet = "FAO整理"
    pvt.to_excel(writer, sheet_name=sheet, index=False)
    wb = writer.book
    ws = writer.sheets[sheet]

    header_fmt = wb.add_format({
        "bold": True, "text_wrap": True, "align": "center", "valign": "vcenter"
    })
    year_fmt = wb.add_format({"num_format": "0"})
    yield_fmt = wb.add_format({"num_format": "#,##0.00", "font_color": "#2369E8"})
    area_fmt = wb.add_format({"num_format": "#,##0", "font_color": "#D7373F"})

    ws.set_row(0, 36, header_fmt)
    ws.set_column(0, 0, 8, year_fmt)
    for idx, col in enumerate(pvt.columns[1:], start=1):
        is_yield = str(col).startswith("Yield")
        ws.set_column(idx, idx, 18 if is_yield else 20, yield_fmt if is_yield else area_fmt)

    ws.freeze_panes(1, 1)

print(f"已保存到: {out_path}")
