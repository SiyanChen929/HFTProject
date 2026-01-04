
"""
Factor Backtest Template v2 (Deterministic, Parameterizable)
Adds:
- IC summary with hit ratio
- Portfolio stats with annual return ratio (annualized) & win percent
- Saves cumulative NAV curves (.png)

Input CSV columns expected (at least):
['underlying_symbol','datetime','trading_date','close']
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt

# ----------------------- Config -----------------------

@dataclass
class Config:
    csv_path: str
    out_dir: str = "./outputs_v2"
    datetime_col: str = "datetime"
    trading_date_col: str = "trading_date"
    symbol_col: str = "underlying_symbol"
    price_col: str = "close"
    hold_days: int = 1                     # forward holding period in trading days
    factor_list: List[str] = None          # ["rskew","rsv_up","rsv_up_ratio"]
    ic_method: str = "spearman"            # "spearman" or "pearson"
    winsor_pct: Optional[float] = None     # e.g., 0.01
    standardize: bool = False              # z-score per-day-per-factor
    quantiles: int = 3                     # portfolio grouping buckets
    use_ratio_for_portfolio: bool = True   # include rsv_up_ratio in portfolios
    trading_days_per_year: int = 252       # for annualization

    def __post_init__(self):
        if self.factor_list is None:
            self.factor_list = ["rskew", "rsv_up", "rsv_up_ratio"]
        os.makedirs(self.out_dir, exist_ok=True)

# ----------------------- Factor defs -----------------------

def realized_skewness(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    xc = x - x.mean()
    rv = (xc**2).sum()
    if rv == 0:
        return np.nan
    n = len(xc)
    return np.sqrt(n) * (xc**3).sum() / (rv ** 1.5)

def upside_realized_vol(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    return ((x**2) * (x >= 0)).sum()

def realized_var(x: pd.Series) -> float:
    x = x.dropna()
    return (x**2).sum() if not x.empty else np.nan

def winsorize_series(s: pd.Series, p: float) -> pd.Series:
    if p is None or p <= 0:
        return s
    lo, hi = s.quantile([p, 1-p])
    return s.clip(lower=lo, upper=hi)

def standardize_series(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd and np.isfinite(sd) else s*0.0

# ----------------------- Pipeline -----------------------

def load_and_prep(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path, parse_dates=[cfg.datetime_col, cfg.trading_date_col])
    df = df.sort_values([cfg.symbol_col, cfg.datetime_col]).copy()
    df[cfg.price_col] = pd.to_numeric(df[cfg.price_col], errors="coerce")
    df = df.dropna(subset=[cfg.price_col])
    df["log_price"] = np.log(df[cfg.price_col])
    # intraday log returns per symbol per day
    df["log_ret"] = df.groupby([cfg.symbol_col, df[cfg.trading_date_col].dt.date])["log_price"].diff()
    return df.dropna(subset=["log_ret"])

def compute_daily_factors(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = df.groupby([cfg.symbol_col, df[cfg.trading_date_col].dt.date])["log_ret"]
    daily = g.agg(
        rskew = realized_skewness,
        rsv_up = upside_realized_vol,
        rv = realized_var,
        n = "count",
    ).reset_index().rename(columns={cfg.trading_date_col: "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rsv_up_ratio"] = daily["rsv_up"] / daily["rv"]
    return daily

def compute_forward_returns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    daily_close = df.groupby([cfg.symbol_col, df[cfg.trading_date_col].dt.date])[cfg.price_col].last().reset_index()
    daily_close[cfg.trading_date_col] = pd.to_datetime(daily_close[cfg.trading_date_col])
    daily_close = daily_close.sort_values([cfg.symbol_col, cfg.trading_date_col])
    daily_close["cc_ret"] = daily_close.groupby(cfg.symbol_col)[cfg.price_col].pct_change()
    k = int(cfg.hold_days)
    daily_close["fwd_ret"] = daily_close.groupby(cfg.symbol_col)["cc_ret"].shift(-k)
    return daily_close.rename(columns={cfg.trading_date_col: "date"})[[cfg.symbol_col, "date", "fwd_ret"]]

def _apply_xsec_transforms(day_df: pd.DataFrame, factors: List[str], cfg: Config) -> pd.DataFrame:
    d = day_df.copy()
    for f in factors:
        s = d[f]
        if cfg.winsor_pct:
            s = winsorize_series(s, cfg.winsor_pct)
        if cfg.standardize:
            s = standardize_series(s)
        d[f] = s
    return d

def prepare_factor_panel(cfg: Config) -> pd.DataFrame:
    df = load_and_prep(cfg)
    fac = compute_daily_factors(df, cfg)
    fwd = compute_forward_returns(df, cfg)
    panel = fac.merge(fwd, on=[cfg.symbol_col, "date"], how="left")
    panel = panel.groupby("date", group_keys=False).apply(lambda d: _apply_xsec_transforms(d, cfg.factor_list, cfg))
    return panel

# ----------------------- IC -----------------------

def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    if method == "spearman":
        return x[mask].rank().corr(y[mask].rank())
    elif method == "pearson":
        return x[mask].corr(y[mask])
    else:
        raise ValueError("ic_method must be 'spearman' or 'pearson'")

def ic_per_instrument(panel: pd.DataFrame, factor: str, cfg: Config) -> pd.DataFrame:
    rows = []
    for sym, g in panel.groupby(cfg.symbol_col):
        rows.append((sym, _corr(g[factor], g["fwd_ret"], cfg.ic_method), g[factor].notna().sum()))
    return pd.DataFrame(rows, columns=[cfg.symbol_col, "IC", "N_days"]).sort_values("IC")

def cross_sectional_ic_series(panel: pd.DataFrame, factor: str, cfg: Config) -> pd.Series:
    return panel.groupby("date").apply(lambda d: _corr(d[factor], d["fwd_ret"], cfg.ic_method))

def ic_summary(series: pd.Series, trading_days_per_year: int) -> pd.Series:
    s = series.dropna()
    N = len(s)
    if N == 0:
        return pd.Series(dict(N=0, mean_ic=np.nan, std_ic=np.nan, tstat=np.nan, ic_ir=np.nan, hit_ratio=np.nan))
    mean = s.mean()
    std = s.std(ddof=1)
    tstat = mean / (std / np.sqrt(N)) if std > 0 else np.nan
    icir = mean / std * np.sqrt(trading_days_per_year) if std > 0 else np.nan
    hit_ratio = (s > 0).mean()
    return pd.Series(dict(N=N, mean_ic=mean, std_ic=std, tstat=tstat, ic_ir=icir, hit_ratio=hit_ratio))

# ----------------------- Portfolios -----------------------

def build_daily_portfolios(panel: pd.DataFrame, factor: str, cfg: Config) -> pd.DataFrame:
    q = int(cfg.quantiles)
    def one_day(d: pd.DataFrame) -> pd.Series:
        d = d.copy()
        d = d[d[factor].notna() & d["fwd_ret"].notna()]
        if d.shape[0] < q:
            return pd.Series({"ls": np.nan, "long_only": np.nan})
        ranks = d[factor].rank(method="first")
        d["bucket"] = pd.qcut(ranks, q, labels=False) + 1
        top = d.loc[d["bucket"]==1, "fwd_ret"].mean()
        bot = d.loc[d["bucket"]==q, "fwd_ret"].mean()
        return pd.Series({"ls": top - bot, "long_only": top})
    daily = panel.groupby("date").apply(one_day)
    daily.columns = [f"{factor}_ls", f"{factor}_longonly"]
    return daily

def perf_summary(r: pd.Series, trading_days_per_year: int) -> pd.Series:
    r = r.dropna()
    if r.empty:
        return pd.Series(dict(N=0, mean=np.nan, ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan, win_rate=np.nan))
    cum = (1+r).cumprod()
    ann_ret = cum.iloc[-1]**(trading_days_per_year/len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(trading_days_per_year)
    sharpe = r.mean()/r.std(ddof=1) * np.sqrt(trading_days_per_year) if r.std(ddof=1)>0 else np.nan
    dd = cum/cum.cummax() - 1.0
    win_rate = (r > 0).mean()
    return pd.Series(dict(N=len(r), mean=r.mean(), ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=dd.min(), win_rate=win_rate))

def save_nav_plot(r: pd.Series, out_png: str, title: str):
    s = r.dropna()
    if s.empty:
        return
    nav = (1+s).cumprod()
    plt.figure()
    nav.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
# ----------------------- Time-Series (per-instrument) Backtest -----------------------

def _rolling_percentile(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    out = []
    arr = s.values
    for i in range(len(arr)):
        j0 = max(0, i-window+1)
        w = arr[j0:i+1]
        if len(w) < min_obs:
            out.append(np.nan)
            continue
        # 把当前值在窗口内的百分位（0..1），再线性映射到 -1..+1
        rank = (w <= w[-1]).sum() - 1
        out.append((rank / (len(w)-1)) * 2 - 1 if len(w) > 1 else np.nan)
    return pd.Series(out, index=s.index)

# v2 里已有 save_nav_plot；若不存在则定义一个兜底
try:
    save_nav_plot
except NameError:
    def save_nav_plot(r: pd.Series, out_png: str, title: str):
        s = r.dropna()
        if s.empty:
            return
        nav = (1+s).cumprod()
        plt.figure()
        nav.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("NAV")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

def _ts_build_for_symbol(df_sym: pd.DataFrame,
                         factor: str,
                         direction: int = 1,
                         norm_method: str = "zscore",
                         norm_window: int = 60,
                         min_obs: int = 40,
                         z_enter: float = 0.5,
                         allow_short: bool = True,
                         cost_bps: float = 1.0) -> pd.DataFrame:
    """
    单品种时序信号：滚动标准化（zscore 或百分位）→ 阈值入场/出场（±z_enter）
    当日信号持仓，用于下一交易日 fwd_ret（v2 的 panel 已对齐到 next day）；
    成本 = |Δposition| * bps/1e4
    """
    d = df_sym.sort_values("date").reset_index(drop=True).copy()

    # 归一化打分
    if norm_method == "zscore":
        roll_mean = d[factor].rolling(norm_window, min_periods=min_obs).mean()
        roll_std  = d[factor].rolling(norm_window, min_periods=min_obs).std(ddof=0)
        score_raw = (d[factor] - roll_mean) / roll_std
    elif norm_method == "percentile":
        score_raw = _rolling_percentile(d[factor], norm_window, min_obs)
    else:
        raise ValueError("norm_method must be 'zscore' or 'percentile'")

    score = direction * score_raw

    # 阈值离散化信号
    sig = np.where(score >= z_enter, 1.0,
          np.where(score <= -z_enter, -1.0 if allow_short else 0.0, 0.0)).astype(float)

    # 持仓与成本
    pos = sig
    pos_prev = np.r_[0.0, pos[:-1]]
    turnover = np.abs(pos - pos_prev)
    cost = turnover * (cost_bps/10000.0)

    # 收益（次日）
    gross = pos * d["fwd_ret"].values
    net = gross - cost

    out = d[["date", factor, "fwd_ret"]].copy()
    out["score"] = score
    out["signal"] = sig
    out["position"] = pos
    out["turnover"] = turnover
    out["ret_gross"] = gross
    out["ret_net"] = net
    return out

def _ts_perf(r: pd.Series, tdpy: int = 252) -> dict:
    r = r.dropna()
    if r.empty:
        return dict(N=0, ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan, win_rate=np.nan)
    nav = (1+r).cumprod()
    ann_ret = nav.iloc[-1]**(tdpy/len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(tdpy)
    sharpe = r.mean()/r.std(ddof=1) * np.sqrt(tdpy) if r.std(ddof=1)>0 else np.nan
    dd = nav/nav.cummax() - 1.0
    win_rate = (r > 0).mean()
    return dict(N=len(r), ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=dd.min(), win_rate=win_rate)

def _ts_trade_stats(pos: pd.Series, r_net: pd.Series) -> dict:
    """
    用连续非零持仓段定义一笔交易；统计笔数与胜率（>0 的交易占比）
    """
    p = pos.fillna(0.0).values
    rn = r_net.fillna(0.0).values
    wins = 0; trades = 0; cur = 0.0; in_tr = False
    for i in range(len(p)):
        if not in_tr and p[i] != 0:
            in_tr = True; cur = rn[i]
        elif in_tr and p[i] != 0:
            cur += rn[i]
        elif in_tr and p[i] == 0:
            trades += 1
            wins += 1 if cur > 0 else 0
            in_tr = False; cur = 0.0
    if in_tr:
        trades += 1; wins += 1 if cur > 0 else 0
    return dict(trades=trades, win_rate_trade=(wins/trades if trades>0 else np.nan))

def ts_backtest(panel: pd.DataFrame,
                out_dir: str,
                factors: List[str],
                direction: Dict[str, int] = None,
                norm_method: str = "zscore",
                norm_window: int = 60,
                min_obs: int = 40,
                z_enter: float = 0.5,
                allow_short: bool = True,
                cost_bps: float = 1.0,
                trading_days_per_year: int = 252) -> None:
    """
    对每个品种独立做时序择时，并输出：
      - 每品种逐日明细 CSV + 净值曲线 PNG
      - 每因子逐品种汇总统计 CSV
      - 等权聚合的逐日与汇总
    依赖：panel 中已有 ['underlying_symbol','date', 因子列, 'fwd_ret']
    """
    if direction is None:
        direction = {f: 1 for f in factors}

    for f in factors:
        per_sym_rows = []
        eqw_daily = []

        for sym, g in panel.groupby("underlying_symbol"):
            res = _ts_build_for_symbol(g, f,
                                       direction=direction.get(f, 1),
                                       norm_method=norm_method,
                                       norm_window=norm_window,
                                       min_obs=min_obs,
                                       z_enter=z_enter,
                                       allow_short=allow_short,
                                       cost_bps=cost_bps)
            res.insert(0, "symbol", sym)
            res.to_csv(os.path.join(out_dir, f"TS_{f}_{sym}_daily.csv"), index=False)

            # per-symbol stats
            pstats = _ts_perf(res["ret_net"], tdpy=trading_days_per_year)
            tstats = _ts_trade_stats(res["position"], res["ret_net"])
            pstats.update(tstats)
            pstats.update(dict(symbol=sym))
            per_sym_rows.append(pstats)

            # for equal-weight
            eqw_daily.append(res[["date","ret_net"]].rename(columns={"ret_net": sym}).set_index("date"))

            # nav plot
            save_nav_plot(res.set_index("date")["ret_net"], os.path.join(out_dir, f"ts_nav_{f}_{sym}.png"),
                          f"TS NAV - {f} - {sym}")

        # 汇总表（逐品种）
        per_sym_df = pd.DataFrame(per_sym_rows).set_index("symbol").sort_index()
        per_sym_df.to_csv(os.path.join(out_dir, f"TS_summary_by_instrument_{f}.csv"))

        # 等权聚合
        if eqw_daily:
            eqw = pd.concat(eqw_daily, axis=1).mean(axis=1, skipna=True).rename("ret_net_eqw")
            eqw.index.name = "date"
            eqw.to_frame().to_csv(os.path.join(out_dir, f"TS_{f}_eqw_daily.csv"))

            eqw_stats = _ts_perf(eqw, tdpy=trading_days_per_year)
            pd.Series(eqw_stats).to_csv(os.path.join(out_dir, f"TS_{f}_eqw_summary.csv"))
            save_nav_plot(eqw, os.path.join(out_dir, f"ts_nav_{f}_eqw.png"),
                          f"TS NAV - {f} - Equal-Weight")
# ----------------------- Intraday (5-minute) Factors, IC & Portfolio -----------------------

def _roll_rskew(a: np.ndarray) -> float:
    x = pd.Series(a).dropna()
    if x.empty: return np.nan
    xc = x - x.mean()
    rv = (xc**2).sum()
    if rv == 0: return np.nan
    n = len(xc)
    return np.sqrt(n) * (xc**3).sum() / (rv ** 1.5)

def _roll_rsv_up(a: np.ndarray) -> float:
    x = pd.Series(a).dropna()
    if x.empty: return np.nan
    return ((x**2) * (x >= 0)).sum()

def _roll_rv(a: np.ndarray) -> float:
    x = pd.Series(a).dropna()
    if x.empty: return np.nan
    return (x**2).sum()

def build_intraday_factor_panel(cfg, roll_bars: int = 96, min_bars: int = 24) -> pd.DataFrame:
    """
    读取原始 5m 数据 → 计算 5m 级别的滚动因子（用到“当前时点之前+当前”数据），
    并对齐“下一根 5m 收益”作为预测目标。
    """
    df = pd.read_csv(cfg.csv_path, parse_dates=[cfg.datetime_col, cfg.trading_date_col])
    df = df.sort_values([cfg.symbol_col, cfg.datetime_col]).copy()
    px = pd.to_numeric(df[cfg.price_col], errors="coerce")
    df = df.assign(close=px).dropna(subset=["close"])

    # 5m 收益与对齐
    df["ret_5m"] = df.groupby(cfg.symbol_col)["close"].pct_change()
    df["log_ret_5m"] = np.log1p(df["ret_5m"])
    df["fwd_ret_5m"] = df.groupby(cfg.symbol_col)["ret_5m"].shift(-1)  # 用于下一根
    df["date"] = df[cfg.trading_date_col].dt.date

    # 滚动窗口因子（每个品种独立、无前视）
    def roll_apply(series, func):
        return series.rolling(roll_bars, min_periods=min_bars).apply(func, raw=True)

    by = df.groupby(cfg.symbol_col, group_keys=False)
    df["rskew_5m"] = by["log_ret_5m"].apply(lambda s: roll_apply(s, _roll_rskew))
    df["rsv_up_5m"] = by["log_ret_5m"].apply(lambda s: roll_apply(s, _roll_rsv_up))
    df["rv_5m"] = by["log_ret_5m"].apply(lambda s: roll_apply(s, _roll_rv))
    df["rsv_up_ratio_5m"] = df["rsv_up_5m"] / df["rv_5m"]

    # 清理
    cols = [cfg.symbol_col, cfg.datetime_col, "date", "ret_5m", "fwd_ret_5m",
            "rskew_5m", "rsv_up_5m", "rv_5m", "rsv_up_ratio_5m"]
    return df[cols].dropna(subset=["fwd_ret_5m"])

def _corr_pair(x: pd.Series, y: pd.Series, method: str) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 3: return np.nan
    if method == "spearman":
        return x[m].rank().corr(y[m].rank())
    elif method == "pearson":
        return x[m].corr(y[m])
    else:
        raise ValueError("ic_method must be 'spearman' or 'pearson'")

def ic5m_per_instrument_daily(df_intra: pd.DataFrame, factor_col: str, ic_method: str = "spearman") -> pd.DataFrame:
    """
    对于每个【品种-交易日】：用当日内的所有 5m 样本做相关：factor(5m, t) vs fwd_ret_5m(t)。
    输出：长表（symbol, date, ic）
    """
    out = []
    for sym, gsym in df_intra.groupby("underlying_symbol"):
        for d, gd in gsym.groupby("date"):
            ic = _corr_pair(gd[factor_col], gd["fwd_ret_5m"], ic_method)
            out.append((sym, pd.to_datetime(d), ic, len(gd)))
    return pd.DataFrame(out, columns=["underlying_symbol","date","ic","N_bar_in_day"])

def ic_series_summary(s: pd.Series, periods_per_year: int) -> pd.Series:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return pd.Series(dict(N=0, mean_ic=np.nan, std_ic=np.nan, tstat=np.nan, ic_ir=np.nan, hit_ratio=np.nan))
    mu = s.mean(); sd = s.std(ddof=1)
    tstat = mu / (sd/np.sqrt(n)) if sd>0 else np.nan
    ic_ir = mu / sd * np.sqrt(periods_per_year) if sd>0 else np.nan
    hit = (s > 0).mean()
    return pd.Series(dict(N=n, mean_ic=mu, std_ic=sd, tstat=tstat, ic_ir=ic_ir, hit_ratio=hit))

def estimate_bars_per_day(df_intra: pd.DataFrame) -> int:
    counts = df_intra.groupby(["underlying_symbol","date"]).size()
    return int(np.nanmedian(counts.values)) if len(counts) else 0

def intraday_cs_portfolios(df_intra: pd.DataFrame, factor_col: str, q: int = 3) -> pd.DataFrame:
    """
    5m 截面组合：每个时间点对所有品种按 factor 排序分组，下一根 5m 收益结算。
    返回：时间序列（index=timestamp）含 {factor}_ls, {factor}_longonly 两列。
    """
    def one_ts(g):
        g = g.copy()
        m = g[factor_col].notna() & g["fwd_ret_5m"].notna()
        g = g[m]
        if g.shape[0] < q:
            return pd.Series({f"{factor_col}_ls": np.nan, f"{factor_col}_longonly": np.nan})
        ranks = g[factor_col].rank(method="first")
        g["bucket"] = pd.qcut(ranks, q, labels=False) + 1
        top = g.loc[g["bucket"]==q, "fwd_ret_5m"].mean()
        bot = g.loc[g["bucket"]==1, "fwd_ret_5m"].mean()
        return pd.Series({f"{factor_col}_ls": top - bot, f"{factor_col}_longonly": top})

    rets = df_intra.groupby(cfg.datetime_col).apply(one_ts)
    return rets

def perf_summary_with_periods(r: pd.Series, periods_per_year: int) -> pd.Series:
    r = r.dropna()
    if r.empty:
        return pd.Series(dict(N=0, ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan, win_rate=np.nan))
    nav = (1+r).cumprod()
    ann_ret = nav.iloc[-1]**(periods_per_year/len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = r.mean()/r.std(ddof=1) * np.sqrt(periods_per_year) if r.std(ddof=1)>0 else np.nan
    dd = nav/nav.cummax() - 1.0
    win = (r>0).mean()
    return pd.Series(dict(N=len(r), ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=dd.min(), win_rate=win))

# ----------------------- Run -----------------------

def run(cfg: Config) -> Dict[str, pd.DataFrame]:
    panel = prepare_factor_panel(cfg)
    panel.to_csv(os.path.join(cfg.out_dir, "panel_factors_and_fwd.csv"), index=False)

    # CS-IC daily & summary
    cs_ics = []
    summaries = []
    for f in cfg.factor_list:
        s = cross_sectional_ic_series(panel, f, cfg).rename(f)
        cs_ics.append(s)
        summaries.append(ic_summary(s, cfg.trading_days_per_year).rename(f))
    cs_ics = pd.concat(cs_ics, axis=1)
    cs_ics.to_csv(os.path.join(cfg.out_dir, "CS_IC_daily.csv"))
    ic_stats = pd.concat(summaries, axis=1).T
    ic_stats.to_csv(os.path.join(cfg.out_dir, "CS_IC_summary.csv"))

    # Per-instrument TS-IC (deterministic per symbol)
    for f in cfg.factor_list:
        ic_df = ic_per_instrument(panel, f, cfg)
        ic_df.to_csv(os.path.join(cfg.out_dir, f"IC_per_instrument_{f}.csv"), index=False)

    # Portfolios + stats + NAV plots
    pr_list = []
    for f in cfg.factor_list if cfg.use_ratio_for_portfolio else ["rskew","rsv_up"]:
        pr = build_daily_portfolios(panel, f, cfg)
        pr_list.append(pr)
    port_rets = pd.concat(pr_list, axis=1)
    port_rets.to_csv(os.path.join(cfg.out_dir, "portfolio_daily_returns.csv"))

    perf = pd.concat({c: perf_summary(port_rets[c], cfg.trading_days_per_year) for c in port_rets.columns}, axis=1).T
    perf.to_csv(os.path.join(cfg.out_dir, "portfolio_perf_summary.csv"))

    # Save NAV plots
    for col in port_rets.columns:
        save_nav_plot(port_rets[col], os.path.join(cfg.out_dir, f"nav_{col}.png"), f"Cumulative NAV - {col}")
    # === (新增) 单一品种时序因子回测 ===
    # 如果你观察到 IC 为负，方向可以设为 -1；反之为 +1
    ts_direction = {f: -1 for f in cfg.factor_list}  # 按需改成 +1/-1

    ts_backtest(
        panel=panel,
        out_dir=cfg.out_dir,
        factors=cfg.factor_list,        # 也可以只选 ["rskew","rsv_up"]
        direction=ts_direction,
        norm_method="zscore",           # 或 "percentile"
        norm_window=60,
        min_obs=40,
        z_enter=0.5,
        allow_short=True,
        cost_bps=0,
        trading_days_per_year=cfg.trading_days_per_year
    )
    # ===== Intraday (5m) 因子、IC 与 5m 组合 =====
    intra = build_intraday_factor_panel(cfg, roll_bars=96, min_bars=24)  # 可按需调整窗口
    bars_per_day = estimate_bars_per_day(intra)
    periods_per_year = max(1, cfg.trading_days_per_year * max(1, bars_per_day))

    # 逐因子：单一品种·按日（源自当日 5m 样本）的 IC 序列与统计
    intra_factor_map = {
        "rskew": "rskew_5m",
        "rsv_up": "rsv_up_5m",
        "rsv_up_ratio": "rsv_up_ratio_5m",
    }
    ic_summaries = []
    for f in cfg.factor_list:
        fcol = intra_factor_map[f]
        ic_daily = ic5m_per_instrument_daily(intra, fcol, ic_method=cfg.ic_method)
        ic_daily.to_csv(os.path.join(cfg.out_dir, f"IC5m_daily_{f}.csv"), index=False)

        # 每个品种一行：mean_ic/std_ic/tstat/ic_ir/hit_ratio
        per_sym = (ic_daily.groupby("underlying_symbol")["ic"]
                   .apply(lambda s: ic_series_summary(s, cfg.trading_days_per_year))  # 这里按“交易日”年化
                   .reset_index())
        per_sym.to_csv(os.path.join(cfg.out_dir, f"IC5m_summary_{f}.csv"), index=False)
        ic_summaries.append(per_sym.assign(factor=f))

    if ic_summaries:
        pd.concat(ic_summaries, axis=0, ignore_index=True) \
          .to_csv(os.path.join(cfg.out_dir, "IC5m_summary_all_factors.csv"), index=False)

    # 5m 截面组合（每根 bar 排序 → 下一根收益）
    cs_perf_rows = []
    for f in cfg.factor_list:
        fcol = intra_factor_map[f]
        rets5 = intraday_cs_portfolios(intra, fcol, q=3)   # 标的较少建议 3 分位
        rets5.to_csv(os.path.join(cfg.out_dir, f"横截面5分钟_portfolio_returns_{f}.csv"))

        perf = pd.concat({c: perf_summary_with_periods(rets5[c], periods_per_year) for c in rets5.columns}, axis=1).T
        perf.to_csv(os.path.join(cfg.out_dir, f"横截面5分钟_portfolio_perf_{f}.csv"))

        # 可选：保存 5m 净值曲线（不强制）
        for c in rets5.columns:
            nav = (1+rets5[c].dropna()).cumprod()
            plt.figure()
            nav.plot()
            plt.title(f"5m Cumulative NAV - {f} - {c}")
            plt.xlabel("Timestamp"); plt.ylabel("NAV")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, f"CS5m_nav_{f}_{c}.png"))
            plt.close()

    return {"panel": panel, "cs_ic": cs_ics, "ic_stats": ic_stats, "port": port_rets, "perf": perf}

# ----------------------- Script entry -----------------------
if __name__ == "__main__":
    cfg = Config(
        csv_path="./pirce_5m.csv",   # change to your CSV path
        out_dir="./outputs_v2",
        hold_days=1,
        ic_method="spearman",
        winsor_pct=None,
        standardize=False,
        quantiles=3,
        factor_list=["rskew","rsv_up","rsv_up_ratio"],
        use_ratio_for_portfolio=True,
        trading_days_per_year=252
    )
    run(cfg)
