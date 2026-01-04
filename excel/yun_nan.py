import re
import csv
import time
import datetime as dt
import string
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

YEAR_FROM, YEAR_TO = 2021, 2022

ENTRY_TAG = "https://www.cnfin.com/news-list/index.html?tagname=%E5%92%96%E5%95%A1%E4%BB%B7%E6%A0%BC%E6%8C%87%E6%95%B0"
ENTRY_INDEX = "https://indices.cnfin.com/5047/index.html?idx=0"

ALLOW_DOMAINS = ("indices.cnfin.com", "www.cnfin.com")

# -------- 详情页 URL 识别（含发布日期 YYYYMMDD）--------
DETAIL_INDICES_RE = re.compile(r"^https?://indices\.cnfin\.com/.*/detail/(\d{8})/\d+_1\.html$")
DETAIL_WWW_RE     = re.compile(r"^https?://www\.cnfin\.com/(?:dz-lb|zs-lb)/detail/(\d{8})/\d+_1\.html$")
# 旧模板：/index-xh08/20210429/1984765.shtml
# 旧模板（含可选 a/ 子目录）：
#  - https://indices.cnfin.com/index-xh08/20210429/1984765.shtml
#  - https://indices.cnfin.com/index-xh08/a/20210421/1983812.sh
DETAIL_INDICES_OLD_RE = re.compile(r"^https?://indices\.cnfin\.com/index-[^/]+/(?:a/)?(\d{8})/\d+\.shtml$",re.IGNORECASE)


def match_detail_and_date(u: str):
    m = DETAIL_INDICES_RE.match(u)
    if m:
        return m.group(1), "indices"
    m = DETAIL_WWW_RE.match(u)
    if m:
        return m.group(1), "www"
    m = DETAIL_INDICES_OLD_RE.match(u)   # ← 新增
    if m:
        return m.group(1), "indices_old"
    return None, None

def is_valid_detail_url(u: str) -> bool:
    ymd, _ = match_detail_and_date(u)
    return bool(ymd)


def url_date_in_range(u: str, year_from=YEAR_FROM, year_to=YEAR_TO) -> bool:
    ymd, _ = match_detail_and_date(u)
    if not ymd: return False
    y = int(ymd[:4])
    return year_from <= y <= year_to

# -------- 文本日期抽取（备用，不再作为主日期）--------
DATE_PATS = [
    re.compile(r"（\s*(\d{4})[.\-/年](\d{1,2})[.\-/月](\d{1,2})[日]?\s*[—\-–至到~～]+\s*(\d{1,2})[.\-/月](\d{1,2})[日]?\s*）"),
    re.compile(r"（\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*[—\-–至到~～]+\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*）"),
    re.compile(r"截至\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*当周"),
]

def make_price_pat(label_pat: str):

    num = r"[0-9０-９]+(?:[\.．][0-9０-９]+)?"
    unit = r"(?:公?斤|千克|kg|KG)(?:\s*\(\s*kg\s*\))?"
    verb = r"(?:报(?:收)?(?:为)?|为|达)"

    return re.compile(rf"{label_pat}(?:的)?\s*均价\s*{verb}\s*({num})\s*元\s*[\/／]?\s*{unit}")

RE_PRICE = {
    "精品二级":      make_price_pat(r"(?:精品二级(?:咖啡豆)?)"),
    "精品三级":      make_price_pat(r"(?:精品三级(?:咖啡豆)?)"),
    "优质咖啡":      make_price_pat(r"(?:优质咖啡豆|优质咖啡|优质级(?:咖啡)?(?:豆)?)"),
    "商业一级":      make_price_pat(r"(?:商业一级(?:咖啡豆)?)"),
    "商业二级":      make_price_pat(r"(?:商业二级(?:咖啡豆)?)"),
    "商业三级及以下": make_price_pat(r"(?:商业三级及以下(?:咖啡豆)?)"),
}
# —— 兜底所需：标签别名（尽量全）
LABEL_ALIASES = {
    "精品二级": r"(?:精品二级)(?:咖啡(?:豆)?)?",
    "精品三级": r"(?:精品三级)(?:咖啡(?:豆)?)?",
    "优质咖啡": r"(?:优质(?:级)?(?:咖啡)?(?:豆)?)",
    "商业一级": r"(?:商业一级)(?:咖啡(?:豆)?)?",
    "商业二级": r"(?:商业二级)(?:咖啡(?:豆)?)?",
    "商业三级及以下": r"(?:商业三级(?:及以下)?)(?:咖啡(?:豆)?)?",
}

# 金额+单位（单位改为“可选”）：…元[/公斤|千克|kg] 也可只有“元”
NUM_UNIT_OPT = r"([0-9]+(?:\.[0-9]+)?)\s*元(?:\s*[/／]?\s*(?:公斤|千克|kg))?"

# 超宽松直抓：标签与金额之间最多 300 个任意字符（含换行），不强求“均价/报/为”等动词
def make_ultra_relaxed_pat(label_regex: str) -> re.Pattern:
    return re.compile(
        rf"{label_regex}(?:的)?[\s\S]{{0,300}}?{NUM_UNIT_OPT}",
        flags=re.IGNORECASE | re.DOTALL
    )

ULTRA_PATS = {k: make_ultra_relaxed_pat(v) for k, v in LABEL_ALIASES.items()}

# “就近归属”兜底：先找金额，再在其前方 300 字内找最近标签
NEARBY_LABEL = re.compile("|".join(
    f"(?P<L{i}>{v})" for i, v in enumerate(LABEL_ALIASES.values())
), flags=re.IGNORECASE)
NUM_WITH_UNIT_OPT = re.compile(NUM_UNIT_OPT, flags=re.IGNORECASE)

def get(url, **kwargs):
    resp = requests.get(url, headers=HEADERS, timeout=20, **kwargs)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp

def in_year_range(d: dt.date) -> bool:
    return YEAR_FROM <= d.year <= YEAR_TO
def normalize_text(text: str) -> str:
    # 全角数字/标点 → 半角
    trans = str.maketrans("０１２３４５６７８９．，：；、　", "0123456789.,:;, ")
    text = text.translate(trans)
    # 不可见空白 → 空格
    for ch in ("\u00A0", "\u200B", "\u200C", "\u200D", "\uFEFF"):
        text = text.replace(ch, " ")
    text = text.replace("&nbsp;", " ")
    # 单位归一（可选）
    text = text.replace("千克", "公斤").replace("KG", "kg").replace("Kg", "kg")
    # 连续空白压缩
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return (text.replace("\u3000", " ").replace("\xa0", " ").replace("\u200b", ""))

def parse_date_range(text: str):
    for pat in DATE_PATS:
        m = pat.search(text)
        if not m: continue
        g = m.groups()
        if len(g) == 5:
            y1, m1, d1, m2, d2 = map(int, g)
            y2 = y1 if m2 >= m1 else y1 + 1
            return dt.date(y1, m1, d1), dt.date(y2, m2, d2), dt.date(y2, m2, d2)
        if len(g) == 6:
            y1, m1, d1, y2, m2, d2 = map(int, g)
            return dt.date(y1, m1, d1), dt.date(y2, m2, d2), dt.date(y2, m2, d2)
        if len(g) == 3:
            y, mo, d = map(int, g)
            end = dt.date(y, mo, d)
            return end - dt.timedelta(days=6), end, end
    return None, None, None

def extract_prices(text: str):
    out = {}
    trans = str.maketrans("０１２３４５６７８９．", "0123456789.")
    for k, pat in RE_PRICE.items():
        m = pat.search(text)
        if m:
            val = m.group(1).translate(trans)
            try:
                out[k] = float(val)
            except ValueError:
                pass
    return out
def extract_prices_relaxed(text: str, already=set()):
    out = {}

    # 第一阶段：超宽松直抓（单位可选，允许跨行/插字）
    for k, pat in ULTRA_PATS.items():
        if k in already:  # 已命中则跳过
            continue
        m = pat.search(text)
        if m:
            try:
                out[k] = float(m.group(1))
            except Exception:
                pass

    needed = ["精品二级","精品三级","优质咖啡","商业一级","商业二级","商业三级及以下"]
    if all(k in out or k in already for k in needed):
        return out

    # 第二阶段：金额就近归属（向前回看 300 字）
    for m in NUM_WITH_UNIT_OPT.finditer(text):
        try:
            val = float(m.group(1))
        except Exception:
            continue
        start_idx = m.start()
        window = text[max(0, start_idx - 300): start_idx]

        best = None
        for m2 in NEARBY_LABEL.finditer(window):
            best = m2  # 最后一个（最近）
        if not best:
            continue

        label_text = best.group(0)
        std_key = None
        for k, rx in LABEL_ALIASES.items():
            if re.fullmatch(rx, label_text, flags=re.IGNORECASE):
                std_key = k; break
        if std_key and std_key not in out and std_key not in already:
            out[std_key] = val

    return out

# -------- URL 清洗 & 从 URL 取发布日期 --------
_URL_SAFE = set(string.ascii_letters + string.digits + ":-._~/?#[]@!$&'()*+,;=%")
def normalize_url(u: str) -> str:
    if not isinstance(u, str): return ""
    for ch in ("\ufeff", "\u200b", "\u200e", "\u200f", "\xa0"):
        u = u.replace(ch, "")
    u = "".join(ch for ch in u if ch in _URL_SAFE).strip().rstrip("#")
    return u.replace("HTTP://", "http://").replace("HTTPS://", "https://")

def get_pubdate_from_url(link: str):
    ymd, _ = match_detail_and_date(link)
    if not ymd: return None
    return dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))

# -------- 链接发现：indices 频道页 --------
def find_article_links_from_indices(max_pages=300):
    links = set()
    for page in range(max_pages):
        url = f"{ENTRY_INDEX}&page={page}"
        try:
            html = get(url).text
        except Exception:
            break
        soup = BeautifulSoup(html, "html.parser")
        batch = 0
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if is_valid_detail_url(href) and url_date_in_range(href):
                links.add(href); batch += 1
        if batch == 0 and page > 10:
            break
        time.sleep(0.12)
    return list(links)

# -------- 标签页：采全量 -> 导出 -> 再筛 --------
def harvest_all_tag_links(max_pages=400, dump_csv=True):
    seen, finals = set(), []
    for page in range(max_pages):
        url = ENTRY_TAG + f"&page={page}"
        try:
            resp = get(url)
        except Exception:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        batch = 0
        for a in soup.find_all("a", href=True):
            raw = urljoin(url, a["href"])
            if raw in seen: continue
            seen.add(raw)
            try:
                r2 = get(raw)  # 跟随重定向
                final_url = r2.url
            except Exception:
                continue
            finals.append(final_url); batch += 1
        if batch == 0 and page > 10:
            break
        time.sleep(0.1)
    finals = sorted(set(finals))
    print(f"[TAG] tag页共采集原始链接 {len(seen)} 条；最终URL（去重后）{len(finals)} 条")
    if dump_csv:
        out = "tag_all_links.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["url"])
            for u in finals: w.writerow([u])
        print(f"[TAG] 已导出全量链接清单：{out}")
    return finals

def filter_detail_urls(urls):
    kept, dropped = [], []
    for u in urls:
        if is_valid_detail_url(u) and url_date_in_range(u):
            kept.append(u)
        else:
            # 调试：看看是“形态不匹配”还是“年份不在范围内”
            ymd, src = match_detail_and_date(u)
            reason = "年份不在范围" if ymd and not url_date_in_range(u) else "形态未匹配"
            dropped.append((u, reason))
    print(f"[TAG] 详情页（{YEAR_FROM}–{YEAR_TO}）保留 {len(kept)} 条；其它 {len(dropped)} 条")
    if dropped:
        print("  示例：", dropped[:5])
    return sorted(set(kept))

# -------- 栏目页（dz-lb / zs-lb）--------
def find_article_links_from_www_channels(max_pages=300):
    links = set()
    channels = [
        "https://www.cnfin.com/dz-lb/index.html",
        "https://www.cnfin.com/zs-lb/index.html",
    ]
    for base in channels:
        for page in range(max_pages):
            url = base + (f"?page={page}" if "?" not in base else f"&page={page}")
            try:
                html = get(url).text
            except Exception:
                break
            soup = BeautifulSoup(html, "html.parser")
            batch = 0
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                if is_valid_detail_url(href) and url_date_in_range(href):
                    links.add(href); batch += 1
            if batch == 0 and page > 10:
                break
            time.sleep(0.12)
    return list(links)

def is_pdf_url(u: str) -> bool:
    u = u.lower()
    return u.endswith(".pdf") or ".pdf?" in u

# -------- 文章解析：按“URL发布日期”为主；允许缺列 --------
def parse_article(link: str):
    try:
        resp = get(link)
    except Exception as e:
        print("请求失败:", link, e)
        return None

    text = clean_html(resp.text)
    text = normalize_text(text)  # ← 新增：强力标准化

    pub_dt = get_pubdate_from_url(link)  # 主日期
    # 备用：正文里的周区间（仅校验/参考，不强制）
    _start, _end, _used = parse_date_range(text)

    # 年份过滤基于“发布日期”
    if pub_dt and not in_year_range(pub_dt):
        return None

    prices = extract_prices(text)

    # 若还有缺，尝试超宽松兜底
    missing_before = {k for k in ["精品二级", "精品三级", "优质咖啡", "商业一级", "商业二级", "商业三级及以下"] if
                      k not in prices}
    if missing_before:
        relaxed = extract_prices_relaxed(text, already=set(prices.keys()))
        prices.update(relaxed)

    if not any(k in prices for k in ["精品二级", "精品三级", "优质咖啡", "商业一级", "商业二级", "商业三级及以下"]):
        print("价格不全(全部缺失):", link, prices)
        return None

    row = {
        "日期": pub_dt.strftime("%Y-%m-%d") if pub_dt else "",
        "精品二级": f"{prices['精品二级']:.2f}" if "精品二级" in prices else "",
        "精品三级": f"{prices['精品三级']:.2f}" if "精品三级" in prices else "",
        "优质咖啡": f"{prices['优质咖啡']:.2f}" if "优质咖啡" in prices else "",
        "商业一级": f"{prices['商业一级']:.2f}" if "商业一级" in prices else "",
        "商业二级": f"{prices['商业二级']:.2f}" if "商业二级" in prices else "",
        "商业三级及以下": f"{prices['商业三级及以下']:.2f}" if "商业三级及以下" in prices else "",
        "来源链接": link
    }
    return row

# -------- 从文件读取：多编码 + URL 归一化 + 兜底尝试 --------
def load_links_from_file(path="links.txt"):
    encodings_to_try = ["utf-8", "utf-8-sig", "utf-16", "gb18030", "latin-1"]
    raw_lines = None
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc) as f:
                raw_lines = f.readlines()
            print(f"[FILE] 用 {enc} 成功读取 {len(raw_lines)} 行")
            break
        except UnicodeDecodeError:
            continue
    if raw_lines is None:
        raise RuntimeError("无法解码 links.txt，请确认保存成 UTF-8 纯文本")

    urls = []
    for line in raw_lines:
        u = normalize_url(line)
        if u:
            urls.append(u)
    urls = sorted(set(urls))
    print(f"[FILE] 读取原始链接 {len(urls)} 条")
    return urls

def crawl_from_links_file(path="links.txt"):
    """从 links.txt 逐条解析（不做白名单/形态筛选）
    - 只做：去重、跳过 .pdf
    - 解析：交给 parse_article（它会按 URL 抽发布日期；允许缺列保留）
    """
    all_urls = load_links_from_file(path)

    # 1) 去重 + 排序（不做任何筛选）
    urls_to_parse = sorted(set(all_urls))
    print(f"[FILE] 将直接解析（不筛选）: {len(urls_to_parse)} 条")
    if len(urls_to_parse) > 0:
        print("  示例:", urls_to_parse[:3])

    rows, fail = [], []
    total = len(urls_to_parse)

    for i, link in enumerate(urls_to_parse, start=1):
        print(f"[{i}/{total}] 解析中: {repr(link)}")

        # 极简保护：跳过 PDF
        if is_pdf_url(link):
            print("  -> 跳过 PDF:", link)
            continue

        row = parse_article(link)
        if row:
            rows.append(row)
            # 允许缺列（方案 A），这里只提示“可能含缺失”
            print(f"  -> 成功: 日期={row.get('日期','')}（可能含缺列） 来源={link}")
        else:
            fail.append(link)
            print("  -> 失败（正文无法抽取任何价格）:", link)

        time.sleep(0.1)

    # 2) 去重：按 (日期, 来源链接) 保留唯一一条（防止不同文章互相覆盖）
    by_key = {}
    for r in rows:
        key = (r.get("日期", ""), r.get("来源链接", ""))
        by_key[key] = r
    rows = sorted(by_key.values(), key=lambda x: (x.get("日期", ""), x.get("来源链接", "")))

    # 3) 导出 CSV
    out = "yunnan_coffee_2021_2022.csv"
    cols = ["日期","精品二级","精品三级","优质咖啡","商业一级","商业二级","商业三级及以下","来源链接"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print("\n===== 文件爬取完成 =====")
    print(f"成功解析(去重后): {len(rows)} 条 -> {out}")
    if fail:
        print(f"失败条数: {len(fail)}，示例:", fail[:5])


# -------- 自动发现（保留）--------
def crawl():
    raw1 = find_article_links_from_indices()
    tag_all = harvest_all_tag_links()
    raw2 = filter_detail_urls(tag_all)
    raw3 = find_article_links_from_www_channels()

    raw_links = set(raw1) | set(raw2) | set(raw3)
    kept = sorted([u for u in raw_links if is_valid_detail_url(u) and url_date_in_range(u)])

    from collections import Counter
    src_cnt = Counter(("indices" if "indices.cnfin.com" in u else "www") for u in kept)
    print(f"[INFO] 抓到详情链接（{YEAR_FROM}–{YEAR_TO}）：{len(kept)} 条 -> 来源: {dict(src_cnt)}")
    print("  indices示例:", [u for u in kept if 'indices.cnfin.com' in u][:3])
    print("  www示例:", [u for u in kept if 'www.cnfin.com' in u][:3])

    rows, fail_date, fail_price, pdf_skipped = [], [], [], []
    total = len(kept)
    for i, link in enumerate(kept, start=1):
        if is_pdf_url(link):
            pdf_skipped.append(link); print(f"[{i}/{total}] 跳过 PDF: {link}"); continue
        print(f"[{i}/{total}] 解析中: {link}")
        row = parse_article(link)
        if row:
            rows.append(row); print(f"  -> 成功: {row['日期']} 六列可能含缺失")
        else:
            try:
                text = clean_html(get(link).text)
                if not parse_date_range(text)[1]:
                    fail_date.append(link)
                else:
                    fail_price.append(link)
            except Exception as e:
                print("  -> 回读失败:", e)
        time.sleep(0.15)

    # 去重并导出
    by_key = {(r["日期"], r["来源链接"]): r for r in rows}
    rows = sorted(by_key.values(), key=lambda x: (x["日期"] or "", x["来源链接"]))

    out = "yunnan_coffee_2021_2022.csv"
    cols = ["日期","精品二级","精品三级","优质咖啡","商业一级","商业二级","商业三级及以下","来源链接"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows: writer.writerow(r)

    print("\n===== 爬取完成 =====")
    print(f"成功解析(去重后): {len(rows)} 条 -> {out}")
    print(f"跳过 PDF: {len(pdf_skipped)} 条")
    print(f"失败-日期未识别: {len(fail_date)} 条")
    print(f"失败-价格不全: {len(fail_price)} 条")

if __name__ == "__main__":
    # 单测两条
    crawl_one = lambda u: (print("单测解析：", u), print("解析结果：", parse_article(u)) )
    crawl_one("https://indices.cnfin.com/jgzs/wenzixiangqingye/detail/20220413/3581949_1.html")
    crawl_one("https://indices.cnfin.com/jgzs/wenzixiangqingye/detail/20220420/3588648_1.html")

    # 从文件跑（请用纯文本保存为 links.txt；不要用 .docx）
    crawl_from_links_file("links.txt")
    # 如需自动发现入口：
    # crawl()
