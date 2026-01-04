"""
Factor Backtest - SLIM
目标：最小输出（一个表 + 一张图），自动方向修正（按 mean IC），包含 5m 表现
输入字段至少包含：['underlying_symbol','datetime','trading_date','close']
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- Config ----------------
@dataclass
class Config:
    csv_path: str = "./pirce_5m.csv"
    out_dir: str = "./outputs_slim"
    datetime_col: str = "datetime"
    trading_date_col: str = "trading_date"
    symbol_col: str = "underlying_symbol"
    price_col: str = "close"

    factor_list: List[str] = None          # ["rskew","rsv_up","rsv_up_ratio"]
    ic_method: str = "spearman"            # or "pearson"
    quantiles: int = 3
    trading_days_per_year: int = 252

    # 5m 滚动窗口（用于构造 5m 因子）
    roll_bars_5m: int = 96
    min_bars_5m: int = 24

    # IC 滚动均值用于画图
    ic_roll_window: int = 20

    def __post_init__(self):
        if self.factor_list is None:
            self.factor_list = ["rskew", "rsv_up", "rsv_up_ratio"]
        os.makedirs(self.out_dir, exist_ok=True)


# ---------------- Helpers ----------------
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

def realized_skewness(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty: return np.nan
    xc = x - x.mean()
    rv = (xc**2).sum()
    if rv == 0: return np.nan
    n = len(xc)
    return np.sqrt(n) * (xc**3).sum() / (rv ** 1.5)

def upside_realized_vol(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty: return np.nan
    return ((x**2) * (x >= 0)).sum()

def realized_var(x: pd.Series) -> float:
    x = x.dropna()
    return (x**2).sum() if not x.empty else np.nan

def corr_xy(x: pd.Series, y: pd.Series, method: str) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 3: return np.nan
    if method == "spearman":
        return x[m].rank().corr(y[m].rank())
    return x[m].corr(y[m])

def perf_summary(r: pd.Series, periods_per_year: int) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return dict(N=0, ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan, win_rate=np.nan)
    nav = (1 + r).cumprod()
    ann_ret = nav.iloc[-1] ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year) if r.std(ddof=1) > 0 else np.nan
    dd = nav / nav.cummax() - 1
    return dict(N=len(r), ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=dd.min(), win_rate=(r > 0).mean())


# ---------------- Daily factor panel ----------------
def build_daily_panel(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path, parse_dates=[cfg.datetime_col, cfg.trading_date_col]).copy()
    df = df.sort_values([cfg.symbol_col, cfg.datetime_col])
    df[cfg.price_col] = pd.to_numeric(df[cfg.price_col], errors="coerce")
    df = df.dropna(subset=[cfg.price_col])

    df["log_price"] = np.log(df[cfg.price_col])
    df["log_ret_5m"] = df.groupby([cfg.symbol_col, df[cfg.trading_date_col].dt.date])["log_price"].diff()

    g = df.groupby([cfg.symbol_col, df[cfg.trading_date_col].dt.date])["log_ret_5m"]
    daily = g.agg(
        rskew=realized_skewness,
        rsv_up=upside_realized_vol,
        rv=realized_var,
        n="count",
    ).reset_index().rename(columns={cfg.trading_date_col: "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rsv_up_ratio"] = daily["rsv_up"] / daily["rv"]

    # close-to-close 日收益 + 对齐到 next day
    close = df.groupby([cfg.symbol_col, df[cfg.trading_date_col].dt.date])[cfg.price_col].last().reset_index()
    close[cfg.trading_date_col] = pd.to_datetime(close[cfg.trading_date_col])
    close = close.sort_values([cfg.symbol_col, cfg.trading_date_col])
    close["ret_1d"] = close.groupby(cfg.symbol_col)[cfg.price_col].pct_change()
    close["fwd_ret_1d"] = close.groupby(cfg.symbol_col)["ret_1d"].shift(-1)
    close = close.rename(columns={cfg.trading_date_col: "date"})

    panel = daily.merge(close[[cfg.symbol_col, "date", "fwd_ret_1d"]], on=[cfg.symbol_col, "date"], how="left")
    return panel.sort_values(["date", cfg.symbol_col])


def cs_ic_series(panel: pd.DataFrame, factor: str, cfg: Config) -> pd.Series:
    return (
        panel.groupby("date", group_keys=False)[[factor, "fwd_ret_1d"]]
             .apply(lambda d: corr_xy(d[factor], d["fwd_ret_1d"], cfg.ic_method))
    )


def build_daily_cs_port(panel: pd.DataFrame, factor: str, sign: int, cfg: Config) -> pd.DataFrame:
    """按因子(已乘方向sign)排序做等权多空/多头，t->t+1"""
    q = int(cfg.quantiles)
    fac = sign * panel[factor]
    def one_day(d: pd.DataFrame) -> pd.Series:
        m = fac.loc[d.index].notna() & d["fwd_ret_1d"].notna()
        if m.sum() < q:
            return pd.Series({"ls": np.nan, "lo": np.nan})
        ranks = fac.loc[d.index][m].rank(method="first")
        buckets = pd.qcut(ranks, q, labels=False) + 1
        top = d.loc[m].loc[buckets == q, "fwd_ret_1d"].mean()
        bot = d.loc[m].loc[buckets == 1, "fwd_ret_1d"].mean()
        return pd.Series({"ls": top - bot, "lo": top})
    ret = panel.groupby("date").apply(one_day)
    ret.columns = [f"{factor}_ls", f"{factor}_lo"]
    return ret


# ---------------- 5m intraday portfolios ----------------
def build_5m_factors_and_returns(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path, parse_dates=[cfg.datetime_col, cfg.trading_date_col]).copy()
    df = df.sort_values([cfg.symbol_col, cfg.datetime_col])
    df[cfg.price_col] = pd.to_numeric(df[cfg.price_col], errors="coerce")
    df = df.dropna(subset=[cfg.price_col])

    df["ret_5m"] = df.groupby(cfg.symbol_col)[cfg.price_col].pct_change()
    df["log_ret_5m"] = np.log1p(df["ret_5m"])
    df["fwd_ret_5m"] = df.groupby(cfg.symbol_col)["ret_5m"].shift(-1)
    df["date"] = df[cfg.trading_date_col].dt.date

    def roll_apply(s, func, w, m):
        return s.rolling(w, min_periods=m).apply(func, raw=True)

    by = df.groupby(cfg.symbol_col, group_keys=False)
    df["rskew_5m"] = by["log_ret_5m"].apply(lambda s: roll_apply(s, lambda a: realized_skewness(pd.Series(a)), cfg.roll_bars_5m, cfg.min_bars_5m))
    df["rsv_up_5m"] = by["log_ret_5m"].apply(lambda s: roll_apply(s, lambda a: upside_realized_vol(pd.Series(a)), cfg.roll_bars_5m, cfg.min_bars_5m))
    df["rv_5m"] = by["log_ret_5m"].apply(lambda s: roll_apply(s, lambda a: realized_var(pd.Series(a)), cfg.roll_bars_5m, cfg.min_bars_5m))
    df["rsv_up_ratio_5m"] = df["rsv_up_5m"] / df["rv_5m"]

    return df.dropna(subset=["fwd_ret_5m"])


def build_5m_cs_port(intra: pd.DataFrame, factor_5m: str, sign: int, cfg: Config) -> pd.DataFrame:
    """每根5m截面排序 → 下一根收益"""
    q = int(cfg.quantiles)
    def one_ts(g: pd.DataFrame) -> pd.Series:
        f = sign * g[factor_5m]
        m = f.notna() & g["fwd_ret_5m"].notna()
        if m.sum() < q:
            return pd.Series({f"{factor_5m}_ls": np.nan, f"{factor_5m}_lo": np.nan})
        ranks = f[m].rank(method="first")
        buckets = pd.qcut(ranks, q, labels=False) + 1
        top = g.loc[m].loc[buckets == q, "fwd_ret_5m"].mean()
        bot = g.loc[m].loc[buckets == 1, "fwd_ret_5m"].mean()
        return pd.Series({f"{factor_5m}_ls": top - bot, f"{factor_5m}_lo": top})
    return intra.groupby(cfg.datetime_col).apply(one_ts)
# ===================== Per-instrument IC (long) & TS performance =====================

def _ic_per_instrument_long(panel: pd.DataFrame, factor_list: List[str], tdpy: int,
                            ic_method: str, ret_col: str = "fwd_ret_1d") -> pd.DataFrame:
    out = []
    for f in factor_list:
        for sym, g in panel.groupby("underlying_symbol"):
            if ret_col not in g.columns:
                continue
            s = g[[f, ret_col]].dropna()
            if s.empty:
                N = 0; mean_ic = std_ic = tstat = icir = hit = np.nan
            else:
                N = len(s)
                ic_val = _corr(s[f], s[ret_col], ic_method)  # 单品种TS-IC
                mean_ic = ic_val
                std_ic = np.nan
                tstat = np.nan
                icir = np.nan
                hit = np.nan
            out.extend([
                (sym, "N",        float(N), f),
                (sym, "mean_ic",  float(mean_ic) if np.isfinite(mean_ic) else np.nan, f),
                (sym, "std_ic",   float(std_ic) if np.isfinite(std_ic) else np.nan, f),
                (sym, "tstat",    float(tstat) if np.isfinite(tstat) else np.nan, f),
                (sym, "ic_ir",    float(icir) if np.isfinite(icir) else np.nan, f),
                (sym, "hit_ratio",float(hit) if np.isfinite(hit) else np.nan, f),
            ])
    return pd.DataFrame(out, columns=["underlying_symbol","level_1","ic","factor"])


def _ts_perf_from_panel(panel: pd.DataFrame,
                        factor_list: List[str],
                        tdpy: int,
                        ic_method: str,
                        ret_col: str = "fwd_ret_1d",
                        norm_window: int = 60,
                        min_obs: int = 40,
                        z_enter: float = 0.5,
                        allow_short: bool = True,
                        cost_bps: float = 0.0) -> pd.DataFrame:
    """
    每个（品种×因子）的日频时序策略：
      - 因子滚动zscore → 分数 * sign
      - 分数 >= z_enter 做多；<= -z_enter 做空（allow_short=False 则空仓）
      - position_t 乘以 next-day ret_col 计入；成本=|Δpos|*bps/1e4
      - 方向 sign 由该（品种×因子）的 TS-IC（factor vs ret_col）符号决定
    输出：underlying_symbol, factor, N, ann_ret, ann_vol, sharpe, max_dd, win_rate, sign_used
    """
    rows = []
    for f in factor_list:
        for sym, g in panel.groupby("underlying_symbol"):
            g = g.sort_values("date").reset_index(drop=True)
            if f not in g.columns or ret_col not in g.columns:
                continue
            # 方向：TS-IC 符号
            ic_val = _corr(g[f], g[ret_col], ic_method)
            sign = 1 if (ic_val is np.nan or not np.isfinite(ic_val)) else (1 if ic_val >= 0 else -1)

            # zscore
            mu = g[f].rolling(norm_window, min_periods=min_obs).mean()
            sd = g[f].rolling(norm_window, min_periods=min_obs).std(ddof=0)
            score = sign * (g[f] - mu) / sd

            # 信号
            sig = np.where(score >= z_enter, 1.0,
                  np.where(score <= -z_enter, -1.0 if allow_short else 0.0, 0.0))

            pos = pd.Series(sig, index=g.index)
            pos_prev = pos.shift(1).fillna(0.0)
            turnover = (pos - pos_prev).abs()
            cost = turnover * (cost_bps/10000.0)

            ret_net = pos * g[ret_col] - cost

            # 绩效统计（用你 slim 里的 perf_summary 逻辑）
            stats = perf_summary(ret_net, tdpy)
            rows.append(dict(underlying_symbol=sym, factor=f, sign_used=sign, **stats))
    return pd.DataFrame(rows)



# ---------------- Main ----------------
def run(cfg: Config):
    # 1) 日频因子 & CS-IC（按因子）
    panel = build_daily_panel(cfg)
    cs_ic = {}
    for f in cfg.factor_list:
        cs_ic[f] = cs_ic_series(panel, f, cfg)
    cs_ics = pd.DataFrame(cs_ic)  # 每列一个因子

    # 2) 按 mean IC 自动决定方向（mean>0 → 多高因子；mean<0 → 多低因子）
    sign_map = {f: (1 if cs_ics[f].mean(skipna=True) >= 0 else -1) for f in cfg.factor_list}

    # 3) 日频组合：为每个因子生成 LS/LO，再做“等权 across factors”的汇总
    daily_list = []
    for f in cfg.factor_list:
        daily_list.append(build_daily_cs_port(panel, f, sign_map[f], cfg))
    daily_ret = pd.concat(daily_list, axis=1)
    # 等权因子组合（跨因子等权平均）
    daily_eq_ls = daily_ret[[f"{f}_ls" for f in cfg.factor_list if f"{f}_ls" in daily_ret.columns]].mean(axis=1, skipna=True).rename("daily_eq_ls")
    daily_eq_lo = daily_ret[[f"{f}_lo" for f in cfg.factor_list if f"{f}_lo" in daily_ret.columns]].mean(axis=1, skipna=True).rename("daily_eq_lo")

    # 4) 5m 因子 & 5m 组合（等权 across factors）
    intra = build_5m_factors_and_returns(cfg)
    bars_per_day = int(intra.groupby(["underlying_symbol","date"]).size().median()) if len(intra)>0 else 0
    per_year_5m = max(1, cfg.trading_days_per_year * max(1, bars_per_day))

    f5map = {"rskew":"rskew_5m","rsv_up":"rsv_up_5m","rsv_up_ratio":"rsv_up_ratio_5m"}
    rets_5m = []
    for f in cfg.factor_list:
        f5 = f5map[f]
        rets_5m.append(build_5m_cs_port(intra, f5, sign_map[f], cfg))
    rets_5m = pd.concat(rets_5m, axis=1)
    m5_eq_ls = rets_5m[[f"{f5map[f]}_ls" for f in cfg.factor_list if f"{f5map[f]}_ls" in rets_5m.columns]].mean(axis=1, skipna=True).rename("m5_eq_ls")
    m5_eq_lo = rets_5m[[f"{f5map[f]}_lo" for f in cfg.factor_list if f"{f5map[f]}_lo" in rets_5m.columns]].mean(axis=1, skipna=True).rename("m5_eq_lo")

    # 5) 汇总表（每个因子一行 + “ALL”等权一行）
    rows = []
    for f in cfg.factor_list:
        row = {"factor": f, "sign_used": sign_map[f],
               "ic_N": cs_ics[f].count(), "ic_mean": cs_ics[f].mean(), "ic_std": cs_ics[f].std(ddof=1)}
        # tstat / icir / hit
        if row["ic_std"] and np.isfinite(row["ic_std"]):
            tstat = row["ic_mean"] / (row["ic_std"] / np.sqrt(max(1, row["ic_N"])))
            icir = row["ic_mean"] / row["ic_std"] * np.sqrt(cfg.trading_days_per_year)
        else:
            tstat, icir = np.nan, np.nan
        row["ic_tstat"] = tstat
        row["ic_icir"] = icir
        row["ic_hit"] = (cs_ics[f] > 0).mean()

        # 该因子对应的 LS/LO（不再单独输出CSV）
        d_ls = daily_ret.get(f"{f}_ls")
        d_lo = daily_ret.get(f"{f}_lo")
        # 5m 不分因子，只做等权，但也可给出单因子表现（如需可扩展）

        # 写入表现（日频，以252年化；5m，以 per_year_5m 年化）
        if d_ls is not None:
            row.update({f"daily_ls_{k}": v for k, v in perf_summary(d_ls, cfg.trading_days_per_year).items()})
        if d_lo is not None:
            row.update({f"daily_lo_{k}": v for k, v in perf_summary(d_lo, cfg.trading_days_per_year).items()})
        rows.append(row)

    # “ALL” 等权 across factors（表现 + IC 平均）
    all_ic = cs_ics.mean(axis=1, skipna=True)
    all_row = {"factor":"ALL", "sign_used": np.nan,
               "ic_N": all_ic.count(), "ic_mean": all_ic.mean(), "ic_std": all_ic.std(ddof=1)}
    if all_row["ic_std"] and np.isfinite(all_row["ic_std"]):
        all_row["ic_tstat"] = all_row["ic_mean"] / (all_row["ic_std"] / np.sqrt(max(1, all_row["ic_N"])))
        all_row["ic_icir"] = all_row["ic_mean"] / all_row["ic_std"] * np.sqrt(cfg.trading_days_per_year)
    else:
        all_row["ic_tstat"] = np.nan; all_row["ic_icir"] = np.nan
    all_row["ic_hit"] = (all_ic > 0).mean()
    # 等权组合表现
    all_row.update({f"daily_eq_ls_{k}": v for k, v in perf_summary(daily_eq_ls, cfg.trading_days_per_year).items()})
    all_row.update({f"daily_eq_lo_{k}": v for k, v in perf_summary(daily_eq_lo, cfg.trading_days_per_year).items()})
    all_row.update({f"m5_eq_ls_{k}": v for k, v in perf_summary(m5_eq_ls, per_year_5m).items()})
    all_row.update({f"m5_eq_lo_{k}": v for k, v in perf_summary(m5_eq_lo, per_year_5m).items()})
    rows.append(all_row)

    master = pd.DataFrame(rows).set_index("factor")
    master.to_csv(os.path.join(cfg.out_dir, "master_summary.csv"))

    # 6) 一张图：等权的 NAV（Daily & 5m，LS/LO） + 等权 IC 序列（及其滚动）
    plt.figure(figsize=(11, 6))
    ax1 = plt.subplot(2,1,1)
    if not daily_eq_ls.dropna().empty: ((1+daily_eq_ls).cumprod()).plot(ax=ax1, label="Daily EQ LS")
    if not daily_eq_lo.dropna().empty: ((1+daily_eq_lo).cumprod()).plot(ax=ax1, label="Daily EQ LO")
    if not m5_eq_ls.dropna().empty:    ((1+m5_eq_ls).cumprod()).plot(ax=ax1, label="5m EQ LS")
    if not m5_eq_lo.dropna().empty:    ((1+m5_eq_lo).cumprod()).plot(ax=ax1, label="5m EQ LO")
    ax1.set_title("Cumulative NAV (Equal-weight across factors)")
    ax1.set_ylabel("NAV"); ax1.legend(loc="best")

    ax2 = plt.subplot(2,1,2)
    all_ic.plot(ax=ax2, alpha=0.5, label="Daily IC (EQ)")
    all_ic.rolling(cfg.ic_roll_window, min_periods=max(5,cfg.ic_roll_window//2)).mean().plot(ax=ax2, label=f"IC MA({cfg.ic_roll_window})", linewidth=2)
    ax2.axhline(0, linewidth=0.8, color="black")
    ax2.set_title("Daily IC (Equal-weight)"); ax2.set_ylabel("IC"); ax2.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "overview.png"))
    plt.close()
    # ===== 分品种 IC（长表）+ 分品种 TS 收益（宽表） =====
    ic_long = _ic_per_instrument_long(panel, cfg.factor_list, cfg.trading_days_per_year, cfg.ic_method,ret_col="fwd_ret_1d")
    ic_long.to_csv(os.path.join(cfg.out_dir, "per_instrument_ic_long.csv"), index=False)

    ts_perf = _ts_perf_from_panel(panel, cfg.factor_list, cfg.trading_days_per_year, cfg.ic_method,
                                  ret_col="fwd_ret_1d",
                                  norm_window=60, min_obs=40, z_enter=0.5,
                                  allow_short=True, cost_bps=0.0)

    ts_perf.to_csv(os.path.join(cfg.out_dir, "per_instrument_ts_perf.csv"), index=False)

    # 一个 Excel 打包（两个 sheet）
    with pd.ExcelWriter(os.path.join(cfg.out_dir, "per_instrument_report.xlsx")) as writer:
        ic_long.to_excel(writer, sheet_name="ic", index=False)
        ts_perf.to_excel(writer, sheet_name="ts_perf", index=False)

    return master


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
