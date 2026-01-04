import re, os, csv, json, time, html, random, pathlib
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============== 基本配置 =====================
BASE      = "https://www.fao.com.cn"
LIST_TPL  = BASE + "/history/117_1511_{year}_{page}.html"
ARTICLE_RE = re.compile(r"^/art/[^/]+\.htm$")
PARM_RE    = re.compile(r"GetPalmReportTable\?parm=([^&\"'>]+)")
API_TPL    = BASE + "/File/GetPalmReportTable?parm={parm}&callback=jsonpcallback"

# 你的登录 cookie（示例，务必换成自己的最新 Cookie）
COOKIE_STR = (
    "325277111passWord=lx7cGWESmRNemwOEAN5LNw%3d%3d; "
    "LoginUserName=MTg2ODAyMDI3MjA%3d; "
    "LoginUserPwd=ODM0a3U2Mjg%3d; "
    "UserID=fJjxr4x4567uUjAF4567DxiNTulA%3d%3d; "
    "UserName=gfqh; "
    "UserGroup=2; "
    "vcode=vrtntA1WWcjNjSGJDc0123zyA%3d%3d"
)

# 输出目录
OUTDIR = pathlib.Path("fao_tables")
OUTDIR.mkdir(exist_ok=True)

# ================= Session & 请求工具 =================
def make_session(cookies_str: str) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                    status_forcelist=(500, 502, 503, 504), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    # 常用头
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    for part in cookies_str.split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            s.cookies.set(k.strip(), v.strip(), domain=".fao.com.cn")
    return s

session = make_session(COOKIE_STR)

def human_sleep(base=1.0, spread=0.8):
    time.sleep(base + random.uniform(0, spread))

def get_with_backoff(url, headers=None, timeout=20, max_attempts=6, base_sleep=1.0):
    attempt = 0
    while True:
        attempt += 1
        resp = session.get(url, headers=headers or {}, timeout=timeout)
        if resp.status_code in (200, 304):
            return resp
        if resp.status_code in (403, 429, 503):
            retry_after = resp.headers.get("Retry-After")
            sleep = int(retry_after) if retry_after and retry_after.isdigit() \
                else base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.8)
            if attempt >= max_attempts:
                resp.raise_for_status()
            time.sleep(sleep);  continue
        if attempt >= max_attempts:
            resp.raise_for_status()
        time.sleep(base_sleep + random.uniform(0, 0.5))

# ================= 抓取步骤 =================
def list_article_links(year, page):
    url = LIST_TPL.format(year=year, page=page)
    r = get_with_backoff(url, headers={"Referer": BASE})
    soup = BeautifulSoup(r.text, "lxml")
    links = [urljoin(BASE, a["href"]) for a in soup.select("a[href]") if ARTICLE_RE.match(a["href"])]
    human_sleep()
    return sorted(set(links))

DATE_PAT = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\b")

def extract_meta(article_url, referer):
    """
    返回 (parm, date_text, title)
    """
    r = get_with_backoff(article_url, headers={"Referer": referer})
    html_text = r.text
    # parm
    m = PARM_RE.search(html_text)
    parm = m.group(1) if m else None
    # 日期（优先找完整 'YYYY-MM-DD HH:MM'，否则 'YYYY-MM-DD'）
    soup = BeautifulSoup(html_text, "lxml")
    text_all = soup.get_text(" ", strip=True)
    dm = DATE_PAT.search(text_all)
    date_text = dm.group(1) if dm else ""
    # 标题（可选）
    title = (soup.select_one("h1") or soup.title or "").get_text(strip=True) if soup else ""
    human_sleep()
    return parm, date_text, title

def fetch_table_payload(parm, article_url):
    api = API_TPL.format(parm=parm)
    r = get_with_backoff(api, headers={"Referer": article_url})
    raw = re.sub(r"^jsonpcallback\(|\)\s*;?\s*$", "", r.text.strip())
    human_sleep()
    return json.loads(raw)

# ================= 表格解析与保存 =================
def _to_number(x):
    if x is None: return None
    s = str(x).strip().replace(",", "")
    if s in ("-", ""): return None
    m = re.match(r"^-?\d+(?:\.\d+)?$", s)
    if m:
        return int(s) if s.isdigit() or (s.startswith("-") and s[1:].isdigit()) else float(s)
    return s

def extract_tables_with_date(payload: dict, date_text: str):
    """
    从 payload["data"] 的 HTML 片段提取所有 <table>，
    并在每张表加入一列 '日期'（表头加上“日期”，数据行填 date_text）。
    返回: [rows, rows, ...]
    """
    snippet = html.unescape(payload["data"])
    soup = BeautifulSoup(snippet, "lxml")
    tables = []
    for t in soup.find_all("table"):
        rows = []
        for tr in t.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if cells:
                rows.append([_to_number(c) for c in cells])
        if rows:
            # 在表头加 '日期'
            rows[0].append("日期")
            # 数据行追加日期值
            for i in range(1, len(rows)):
                rows[i].append(date_text)
            tables.append(rows)
    return tables

def save_tables(tables, year, page, idx, title=""):
    safe_id = f"{year}_{page}_{idx}"
    if title:
        title_sn = re.sub(r"[^\w\u4e00-\u9fa5-]+", "_", title)[:30]
        safe_id += f"_{title_sn}"
    for i, rows in enumerate(tables, 1):
        fn = OUTDIR / f"{safe_id}_table{i}.csv"
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        print(f"✔ Saved {fn}")

# ================= 主流程 =================
def crawl(years=range(2024, 2026), pages=(1, 2)):
    for y in years:
        for p in pages:
            links = list_article_links(y, p)
            print(f"Year {y} page {p} → {len(links)} articles")
            # 可随机打乱，降低模式化
            random.shuffle(links)
            for i, url in enumerate(links, 1):
                try:
                    print("  →", url)
                    parm, date_text, title = extract_meta(url, LIST_TPL.format(year=y, page=p))
                    if not parm:
                        print("   ⚠️ 未找到 parm，跳过")
                        continue
                    payload = fetch_table_payload(parm, url)
                    tables = extract_tables_with_date(payload, date_text)
                    save_tables(tables, y, p, i, title=title)
                    # 小憩，避免触发风控
                    if i % 5 == 0:
                        time.sleep(4 + random.uniform(0, 2))
                except Exception as e:
                    print("   ✖ 失败：", e)
            time.sleep(6 + random.uniform(0, 3))

if __name__ == "__main__":
    crawl(years=range(2023,2024), pages=(1, 2))