
import logging
from io import StringIO
from pathlib import Path
import pandas as pd
import numpy as np

# ========= 配置 =========
DATA_PATH = "/Users/hitomebore/Downloads/PythonProject/tick_data.csv"
SPREAD_TH = 1                         # 触发阈值：卖一-买一 < 1
QTY = 1                                  # 下单手数
ONLY_WHEN_FLAT = False                    # 仅在空仓时买入，避免无限加仓
LOG_FILE = "trade.log"                   # 日志文件
TRADES_CSV = "trades.csv"                # 交易流水文件
RESULTS_SUMMARY = "results.txt"          # 结果摘要

# ========= 日志设置 =========
logger = logging.getLogger("trade")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)


def to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    # 1) 读数据
    df = pd.read_csv(DATA_PATH)
    # 2) 必要列
    need_cols = ["TRADINGTIME", "SELLPRICE01", "BUYPRICE01", "LASTPRICE"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要列：{missing}")

    # 3) 类型与排序
    df["TRADINGTIME"] = pd.to_datetime(df["TRADINGTIME"], errors="coerce")
    df = to_num(df, ["SELLPRICE01", "BUYPRICE01", "LASTPRICE"])
    df = df.sort_values("TRADINGTIME").reset_index(drop=True)

    # 4) 状态
    cash = 0.0
    position = 0
    avg_cost = 0.0
    trades = []

    # 5) 用 (t-1) 触发、在 t 判定
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        t_prev = prev["TRADINGTIME"]
        ask_prev = prev["SELLPRICE01"]
        bid_prev = prev["BUYPRICE01"]
        if np.isnan(ask_prev) or np.isnan(bid_prev):
            continue

        spread_prev = ask_prev - bid_prev

        # 在上一个 tick（t-1）是否产生买入挂单？
        if spread_prev < SPREAD_TH and (not ONLY_WHEN_FLAT or position == 0):
            order_price = float(bid_prev)      # 上一个tick的买一价 = 挂单价
            order_qty = QTY

            # 在当前 tick（t）进行价格撮合判定
            last_curr = curr["LASTPRICE"]
            bid_curr  = curr["BUYPRICE01"]

            cond_last = (pd.notna(last_curr) and last_curr <= order_price)  # 最新成交价 <= 上一tick挂单价
            cond_bid  = (pd.notna(bid_curr)  and bid_curr  <  bid_prev)     # 现买一 < 上一tick的买一

            success = cond_last or cond_bid

            if success:
                fill_price = order_price  # 成交价按挂单价；如需价优可用 min(order_price, last_curr)
                old_pos = position
                position += order_qty
                cash -= fill_price * order_qty
                avg_cost = fill_price if old_pos == 0 else (avg_cost*old_pos + fill_price*order_qty) / (old_pos + order_qty)

                reason = "by_last_vs_prev_order" if cond_last else "bid_drop_vs_prev"
                msg = (f"{curr['TRADINGTIME']} | BUY 成功 qty={order_qty} price={fill_price:.4f} "
                       f"reason={reason} prev_spread={spread_prev:.4f} pos={position} cash={cash:.2f}")
                logger.info(msg)
                trades.append({
                    "time": curr["TRADINGTIME"], "action": "BUY_OK", "qty": order_qty,
                    "price": fill_price, "prev_spread": spread_prev, "reason": reason,
                    "prev_order_price": order_price, "prev_bid": bid_prev, "prev_ask": ask_prev, "t_prev": t_prev,
                    "curr_bid": bid_curr, "curr_last": last_curr,
                    "position": position, "cash": cash
                })
            else:
                msg = (f"{curr['TRADINGTIME']} | BUY 失败（价格条件不满足） "
                       f"prev_order_price={order_price}, curr_last={last_curr}, "
                       f"curr_bid={bid_curr}, prev_bid={bid_prev}, prev_spread={spread_prev:.4f}")
                logger.warning(msg)
                trades.append({
                    "time": curr["TRADINGTIME"], "action": "BUY_FAIL", "qty": order_qty,
                    "price": order_price, "prev_spread": spread_prev,
                    "prev_order_price": order_price, "prev_bid": bid_prev, "prev_ask": ask_prev, "t_prev": t_prev,
                    "curr_bid": bid_curr, "curr_last": last_curr,
                    "reason": "price_conditions_not_met",
                    "position": position, "cash": cash
                })

        # 若上一tick未触发下单，这一tick不做任何判定

    # 6) 期末评估
    last_price = float(df["LASTPRICE"].dropna().iloc[-1]) if df["LASTPRICE"].notna().any() else np.nan
    equity = cash + (position * last_price if not np.isnan(last_price) else 0.0)
    unrealized = (last_price - avg_cost) * position if (position>0 and not np.isnan(last_price)) else 0.0

    # 7) 输出结果
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(TRADES_CSV, index=False, encoding="utf-8")

    summary = [
        f"期末价格(last): {last_price}",
        f"持仓(position): {position}",
        f"均价(avg_cost): {avg_cost:.4f}" if position>0 else "均价(avg_cost): N/A",
        f"现金(cash): {cash:.2f}",
        f"权益(equity): {equity:.2f}",
        f"未实现盈亏(unrealized PnL): {unrealized:.2f}",
        f"交易尝试次数: {len(trades_df)}，成功 {(trades_df['action']=='BUY_OK').sum()} 次，失败 {(trades_df['action']=='BUY_FAIL').sum()} 次"
    ]
    with open(RESULTS_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    logger.info("==== 回测完成 ====")
    for line in summary:
        logger.info(line)

if __name__ == "__main__":
    main()