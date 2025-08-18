import pandas as pd


with open("/Users/hitomebore/Downloads/PythonProject/.venv/INEL2_EC2512_202508.txt", "r", encoding="gb18030") as f:
    text = f.read()

from io import StringIO
df = pd.read_csv(StringIO(text))

# 保存为 csv
df.to_csv("tick_data.csv", index=False, encoding="utf-8")
