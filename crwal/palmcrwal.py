# -*- coding: utf-8 -*-
import re, json, csv, time, pathlib, html
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.fao.com.cn"
LIST_TPL = "https://www.fao.com.cn/history/117_1511_{year}_{page}.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X)",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.fao.com.cn/",
}

# —— 把你贴的两段 Cookie 直接粘到这（建议保存在私密文件/环境变量中）——
RAW_COOKIE_LIST = """325277111passWord=lx7cGWESmRNemwOEAN5LNw%3d%3d; LoginUserName=MTg2ODAyMDI3MjA%3d; LoginUserPwd=ODM0a3U2Mjg%3d; 325277cookieGuid=95f63ed3-55f7-47ec-ad1c-f9df7b2a7193; UserID=fJjxr4x4567uUjAF4567DxiNTulA%3d%3d; UserName=gfqh; UserGroup=2; vcode=vrtntA1WWcjNjSGJDc0123zyA%3d%3d"""
RAW_COOKIE_ART = """325277111passWord=lx7cGWESmRNemwOEAN5LNw%3d%3d; LoginUserName=MTg2ODAyMDI3MjA%3d; LoginUserPwd=ODM0a3U2Mjg%3d; UserID=fJjxr4x4567uUjAF4567DxiNTulA%3d%3d; UserName=gfqh; UserGroup=2; 325277cookieGuid=b02e6ceb-525f-409d-85c4-4ecb6ceabef4; vcode=MLvzCuOfXQh9WQG4567x8K2Pg%3d%3d"""

def make_session(raw_cookie: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    for part in raw_cookie.split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            s.cookies.set(k.strip(), v.strip(), domain=".fao.com.cn")
    return s

sess_list = make_session(RAW_COOKIE_LIST)  # 用于进列表页
sess_art  = make_session(RAW_COOKIE_ART)   # 用于进文章页与表格接口

def list_article_links(year: int, page: int):
    url = LIST_TPL.format(year=year, page=page)
    r = sess_list.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # 只取 /art/*.htm
    links = [urljoin(BASE, a["href"]) for a in soup.select("a[href^='/art/']")]
    return sorted(set(links))

def extract_parm(article_url: str):
    r = sess_art.get(article_url, headers={"Referer": article_url}, timeout=20)
    r.raise_for_status()
    # 在 <script src="...GetPalmReportTable?parm=XXXX&callback=..."></script> 里取 parm
    m = re.search(r"GetPalmReportTable\?parm=([^&\"'>]+)", r.text)
    return m.group(1) if m else None

def fetch_table_payload(parm: str):
    api = f"{BASE}/File/GetPalmReportTable?parm={parm}&callback=jsonpcallback"
    r = sess_art.get(api, headers={"Referer": BASE}, timeout=20)
    r.raise_for_status()
    # 去 JSONP 包装
    raw = re.sub(r"^jsonpcallback\(|\)\s*;?\s*$", "", r.text.strip())
    return json.loads(raw)

def rows_from_payload(payload):
    """
    兼容两种可能：
    1) payload 里直接给了 HTML 片段：如 {"html": "<table>...</table>"} 或 {"Html": "..."} 或 {"tableHtml": "..."}
    2) 结构化：如 {"columns": [...], "rows": [[...],[...]]} / {"data":[...]}
    返回：列表[ [col1,col2,...], ... ]（包含表头）
    """
    # 情况 1：HTML 片段
    html_key = None
    for k in ("html", "Html", "tableHtml", "content", "Content"):
        if isinstance(payload, dict) and k in payload and isinstance(payload[k], str) and "<table" in payload[k]:
            html_key = k; break
    if html_key:
        snippet = payload[html_key]
        snippet = html.unescape(snippet)
        soup = BeautifulSoup(snippet, "lxml")
        table = soup.select_one("table")
        if table:
            rows = []
            for tr in table.select("tr"):
                cells = [c.get_text(strip=True) for c in tr.select("th,td")]
                if cells: rows.append(cells)
            return rows

    # 情况 2：结构化
    if isinstance(payload, dict):
        # 常见字段名猜测
        cols = payload.get("columns") or payload.get("Cols") or payload.get("Header")
        data = payload.get("rows") or payload.get("Rows") or payload.get("Data") or payload.get("data")
        if isinstance(data, list):
            out = []
            if isinstance(cols, list) and cols:
                # 列名可能是字符串或对象数组
                if all(isinstance(c, dict) and "title" in c for c in cols):
                    out.append([c["title"] for c in cols])
                else:
                    out.append([str(c) for c in cols])
            # 行：可能是 list[list] 或 list[dict]
            if data and isinstance(data[0], dict):
                # 用字典键排序（如果有列顺序，用列名对齐）
                keys = [c["field"] if isinstance(c, dict) and "field" in c else c for c in cols] if cols else sorted(data[0].keys())
                out.extend([[row.get(k, "") for k in keys] for row in data])
            else:
                out.extend([[str(x) for x in row] for row in data])
            return out

    # 兜底：把整个 payload 打印成一行
    return [["raw_json"], [json.dumps(payload, ensure_ascii=False)]]

def save_csv(rows, csv_path: pathlib.Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows: writer.writerow(r)

def crawl(years=range(2024, 2026), pages=(1,2), outdir="fao_out"):
    outdir = pathlib.Path(outdir)
    for y in years:
        for p in pages:
            links = list_article_links(y, p)
            print(f"[{y}-{p}] {len(links)} articles")
            for idx, u in enumerate(links, 1):
                try:
                    parm = extract_parm(u)
                    if not parm:
                        print("  - no parm:", u); continue
                    payload = fetch_table_payload(parm)
                    rows = rows_from_payload(payload)
                    # 文件名：年份-页码-序号
                    safe_id = re.sub(r"[^A-Za-z0-9_=+-]", "_", u.rsplit("/",1)[-1].replace(".htm",""))
                    csv_path = outdir / f"{y}_{p:01d}_{idx:02d}_{safe_id}.csv"
                    save_csv(rows, csv_path)
                    print("  - saved:", csv_path.name)
                    time.sleep(0.7)  # 温和限速
                except Exception as e:
                    print("  ! fail:", u, e)
            time.sleep(1.2)

if __name__ == "__main__":
    crawl(years=range(2024, 2026), pages=(1,2), outdir="fao_tables")


