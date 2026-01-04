import matplotlib.pyplot as plt

# 设置中文字体 (macOS)
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 或者 'PingFang HK'
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号

import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Songti SC']   # 或 PingFang HK
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = "yunnan_coffee_2021_2022_dates_filled_again.csv"
df = pd.read_csv(file_path)

# 日期处理
df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
df["年份"] = df["日期"].dt.year
df["月份"] = df["日期"].dt.month
df["日"] = df["日期"].dt.day

# 转换成 "月份+小数" 作为横坐标
df["月份连续"] = df["月份"] + (df["日"] / 31.0)

# 品种列
price_cols = ["精品二级", "精品三级", "优质咖啡", "商业一级", "商业二级"]

# 2018–2022
df_plot = df[df["年份"].between(2018, 2022)]

# 每个品种一张图，不同颜色代表不同年份
for col in price_cols:
    plt.figure(figsize=(10, 6))
    for year, group in df_plot.groupby("年份"):
        plt.plot(group["月份连续"], group[col], label=str(year))
    plt.title(f"{col} - 2018~2022 年月度季节性图")
    plt.xlabel("月份 (1-12)")
    plt.ylabel("价格 (元/公斤)")
    plt.xticks(range(1, 13))
    plt.legend(title="年份")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{col}_2018-2022_季节性图.png", dpi=300)
    plt.close()  # 关闭图表，避免内存占用