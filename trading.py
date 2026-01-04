
import logging
from io import StringIO
from pathlib import Path
import pandas as pd
import numpy as np
import time
start= time.perf_counter()

# ========= 配置 =========
DATA_PATH = "/Users/hitomebore/Downloads/PythonProject/tick_data.csv"
SPREAD_TH = 1                         # 触发阈值：卖一-买一 < 1
QTY = 1                                  # 下单手数
ONLY_WHEN_FLAT =  False                 # 仅在空仓时买入，避免无限加仓
TICKS_AFTER_BUY_TO_SELL = 5             # 买入后第 N 个 tick 才挂卖单
SELL_WAIT_SECONDS = 2.0                 # 卖单最大等待时间窗口（秒）
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

    # 4) 回测状态
    cash = 0.0
    position = 0
    avg_cost = 0.0
    trades = []

    # Pending 卖单（最多一个，简化）
    pending_sell = None  # dict: { 'place_time', 'expire_time', 'order_price', 'qty', 'placed_index' }

    # 5) 遍历 tick
    for i in range(len(df)):
        row = df.iloc[i]
        t = row["TRADINGTIME"]
        ask = row["SELLPRICE01"]
        bid = row["BUYPRICE01"]
        last = row["LASTPRICE"]

        # A) 先处理已有的 pending 卖单的成交/超时判定
        if pending_sell is not None:
            # 如果当前 tick 时间 >= 订单过期时间，则视为这 2 秒内没有任何满足条件的 tick -> 失败
            if i <= pending_sell["placed_index"]:
                pass
            if t >= pending_sell["expire_time"]:
                logger.warning(
                    f"{t} | SELL 失败（2秒窗口内无成交） "
                    f"order_price={pending_sell['order_price']}, placed_at={pending_sell['place_time']}"
                )
                trades.append({
                    "time": t, "action": "SELL_FAIL", "qty": pending_sell["qty"],
                    "price": pending_sell["order_price"], "reason": "timeout_no_matching_tick",
                    "place_time": pending_sell["place_time"], "expire_time": pending_sell["expire_time"],
                })
                pending_sell = None  # 清空
            else:
                # 仍在 2 秒窗口内，检查卖出撮合条件
                order_price = pending_sell["order_price"]
                cond_last = (pd.notna(last) and last >= order_price)
                cond_bid_up = (pd.notna(bid) and bid > order_price)  # 严格大于，避免恒成立

                if cond_last or cond_bid_up:
                    # 卖出成交
                    fill_price = order_price  # 以挂单价成交（可改成 max(order_price, last) 价优）
                    sell_qty = pending_sell["qty"]
                    position -= sell_qty
                    cash += fill_price * sell_qty
                    reason = "by_lastprice_ge_order" if cond_last else "bid_breaks_above_order"

                    logger.info(
                        f"{t} | SELL 成功 qty={sell_qty} price={fill_price:.4f} "
                        f"reason={reason} cash={cash:.2f} pos={position}"
                    )
                    trades.append({
                        "time": t, "action": "SELL_OK", "qty": sell_qty,
                        "price": fill_price, "reason": reason,
                        "place_time": pending_sell["place_time"], "expire_time": pending_sell["expire_time"],
                    })
                    pending_sell = None  # 卖单完成
        # B) 买入触发&成交（用 t-1 的挂单、在 t 判定）
        if i >= 1:
            prev = df.iloc[i-1]
            t_prev = prev["TRADINGTIME"]
            ask_prev = prev["SELLPRICE01"]
            bid_prev = prev["BUYPRICE01"]
            if pd.notna(ask_prev) and pd.notna(bid_prev):
                spread_prev = ask_prev - bid_prev
                # 在上一个 tick 是否触发买入挂单？
                if spread_prev < SPREAD_TH and (not ONLY_WHEN_FLAT or position == 0):
                    # 上一 tick 的买一挂单
                    order_price_buy = float(bid_prev)
                    # 当前 tick 的买入撮合条件（你的规则）
                    cond_last_buy = (pd.notna(last) and last <= order_price_buy)
                    cond_bid_buy  = (pd.notna(bid)  and bid  <  bid_prev)  # 严格小于，避免恒成立
                    success_buy = cond_last_buy or cond_bid_buy

                    if success_buy:
                        # 买入成交
                        fill_buy = order_price_buy
                        old_pos = position
                        position += QTY
                        cash -= fill_buy * QTY
                        avg_cost = fill_buy if old_pos == 0 else (avg_cost*old_pos + fill_buy*QTY)/(old_pos+QTY)

                        reason_buy = "by_last_vs_prev_order" if cond_last_buy else "bid_drop_vs_prev"
                        logger.info(
                            f"{t} | BUY 成功 qty={QTY} price={fill_buy:.4f} "
                            f"reason={reason_buy} pos={position} cash={cash:.2f}"
                        )
                        trades.append({
                            "time": t, "action": "BUY_OK", "qty": QTY,
                            "price": fill_buy, "reason": reason_buy,
                            "prev_time": t_prev, "prev_bid": bid_prev, "prev_ask": ask_prev,
                        })

                        # 安排第 N 个 tick 后放置卖单（若数据不足，则后面不会放单）
                        sell_place_index = i + TICKS_AFTER_BUY_TO_SELL
                        if sell_place_index < len(df):
                            t_place = df.iloc[sell_place_index]["TRADINGTIME"]
                            order_price_sell = float(df.iloc[sell_place_index]["BUYPRICE01"])  # 对手价=当时买一
                            expire_time = t_place + pd.Timedelta(seconds=SELL_WAIT_SECONDS)
                            pending_sell = {
                                "place_time": t_place,
                                "expire_time": expire_time,
                                "order_price": order_price_sell,
                                "qty": QTY,
                                "placed_index": sell_place_index
                            }
                            logger.info(
                                f"{t_place} | 安排 SELL 挂单 对手价={order_price_sell:.4f} "
                                f"等待窗口至 {expire_time}"
                            )
                            trades.append({
                            "time": t_place, "action": "Sell_Planned", "qty": QTY,
                            "price": fill_buy, "reason": reason_buy,
                            "prev_time": t_prev, "prev_bid": bid_prev, "prev_ask": ask_prev,
                        })
                        else:
                            logger.warning(f"{t} | 数据不足，无法在第 {TICKS_AFTER_BUY_TO_SELL} 个 tick 放置卖单")

                    else:
                        logger.warning(
                            f"{t} | BUY 失败（价格条件不满足） "
                            f"prev_order_price={order_price_buy}, last={last}, bid_now={bid}, prev_bid={bid_prev}"
                        )
                        trades.append({
                            "time": t, "action": "BUY_FAIL", "qty": QTY,
                            "price": order_price_buy, "reason": "price_conditions_not_met",
                            "prev_time": t_prev, "prev_bid": bid_prev, "prev_ask": ask_prev,
                        })

        # C) 如果当前 i 正好是卖单的放置点，**不做判定**（放置点本身不判断，判断从“之后的 tick”开始）
        # 上面在处理 pending_sell 时已经保证：只有当 t >= expire_time 才会判定超时

    # 循环结束后，若还有 pending 卖单未到期或未判定，按未成交处理
    if pending_sell is not None:
        logger.warning(
            f"{pending_sell['place_time']} | SELL 失败（数据结束前未在2秒内满足条件） "
            f"order_price={pending_sell['order_price']}"
        )
        trades.append({
            "time": pending_sell["place_time"], "action": "SELL_FAIL", "qty": pending_sell["qty"],
            "price": pending_sell["order_price"], "reason": "eod_no_fill_within_2s",
            "place_time": pending_sell["place_time"], "expire_time": pending_sell["expire_time"],
        })
        pending_sell = None

    # 期末评估
    last_price = float(df["LASTPRICE"].dropna().iloc[-1]) if df["LASTPRICE"].notna().any() else np.nan
    equity = cash + (position * last_price if not np.isnan(last_price) else 0.0)
    unrealized = (last_price - avg_cost) * position if (position>0 and not np.isnan(last_price)) else 0.0

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(TRADES_CSV, index=False, encoding="utf-8")

    summary = [
        f"期末价格(last): {last_price}",
        f"持仓(position): {position}",
        f"均价(avg_cost): {avg_cost:.4f}" if position>0 else "均价(avg_cost): N/A",
        f"现金(cash): {cash:.2f}",
        f"权益(equity): {equity:.2f}",
        f"未实现盈亏(unrealized PnL): {unrealized:.2f}",
        f"记录条数: {len(trades_df)}，BUY_OK={(trades_df['action']=='BUY_OK').sum()}，BUY_FAIL={(trades_df['action']=='BUY_FAIL').sum()}，SELL_OK={(trades_df['action']=='SELL_OK').sum()}，SELL_FAIL={(trades_df['action']=='SELL_FAIL').sum()}"
    ]
    with open(RESULTS_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    logger.info("==== 回测完成 ====")
    for line in summary:
        logger.info(line)

if __name__ == "__main__":
    main()
end=time.perf_counter()
print(f"总运行时间: {end - start:.3f} 秒")