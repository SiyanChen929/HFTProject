import logging
from io import StringIO
from pathlib import Path
import pandas as pd
import time
import numpy as np
start = time.perf_counter()

# ========= 配置 =========
DATA_PATH = "/Users/hitomebore/Downloads/PythonProject/tick_data.csv"
SPREAD_TH = 1.0                         # 触发阈值：卖一-买一 < 1（用于买入触发）
QTY = 1                                 # 下单手数
ONLY_WHEN_FLAT = False                   # 仅在空仓时买入
TICKS_AFTER_BUY_TO_SELL = 5             # 买入后第 N 个 tick 才挂卖单
SELL_WAIT_SECONDS = 2.0                 # 卖单最大等待时间窗口（秒）
LOG_FILE = "trade.log"                  # 日志文件
TRADES_CSV = "trades.csv"               # 交易流水文件
MAX_REPOSTS = 100                   # 同一笔最多重挂次数（含第一次挂为 attempt=1）
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
    # ========= 回测状态 =========
    cash = 0.0
    position = 0
    avg_cost = 0.0
    trades = []

    # 多订单并行：计划队列 + 激活中的卖单
    sell_queue = []  # 每项: {placed_index, place_time, expire_time, order_price, qty}
    active_sells = []  # 每项: {placed_index, place_time, expire_time, order_price, qty}

    # ========= 主循环 =========
    for i in range(len(df)):
        row = df.iloc[i]
        t = row["TRADINGTIME"]
        ask = row["SELLPRICE01"]
        bid = row["BUYPRICE01"]
        last = row["LASTPRICE"]

        # 0) 到点激活：把 sell_queue 中 placed_index == i 的计划，全部转为 active_sells
        if sell_queue:
            # 允许同一根激活多笔
            to_activate = [p for p in sell_queue if p["placed_index"] == i]
            if to_activate:
                # 从队列中移除这些计划
                sell_queue = [p for p in sell_queue if p["placed_index"] != i]
                # 激活到点的 sell 计划
                for plan in to_activate:
                    plan = {**plan, "attempt": 1}  # 第一次挂单
                    active_sells.append(plan)

                    trades.append({
                        "time": plan["place_time"],
                        "action": "SELL_PLANNED",
                        "qty": plan["qty"],
                        "price": plan["order_price"],
                        "place_time": plan["place_time"],
                        "expire_time": plan["expire_time"],
                        "attempt": plan["attempt"],
                    })
                    logger.info(f"{plan['place_time']} | 安排 SELL 挂单 对手价={plan['order_price']:.4f} "
                                f"等待窗口至 {plan['expire_time']} (attempt={plan['attempt']})")

        # A) 并行处理所有激活中的卖单（逐单评估，不影响买入逻辑）
        if active_sells:
            still_active = []
            for od in active_sells:
                place_time = od["place_time"]
                expire_time = od["expire_time"]
                order_price = od["order_price"]

                # 1) 放单当根及任何 t<=place_time 的行：不判定
                if t <= place_time:
                    still_active.append(od)
                    continue

                # 2) 到/跨过过期：撤单并按当前对手价重挂；超过上限则失败
                if t >= expire_time:
                    # 若买一价无效，先不重挂：继续等待下一 tick（不计入 attempt）
                    if pd.isna(bid):
                        still_active.append(od)  # 等下一根再判断
                        continue

                    if od["attempt"] < MAX_REPOSTS:
                        # 记撤单并重挂（时间=当前 tick）
                        trades.append({
                            "time": expire_time,  # 事件发生的“到期时刻”
                            "processed_at": t,  # 实际处理时刻
                            "action": "SELL_CANCEL_REPOST",
                            "qty": od["qty"],
                            "price": od["order_price"],  # 被撤的旧价
                            "new_price": float(bid),  # 以当前买一重挂
                            "place_time": place_time,
                            "expire_time": expire_time,
                            "attempt": od["attempt"],
                        })
                        logger.warning(
                            f"{expire_time} | SELL 撤单并重挂 -> new_price={bid:.4f} "
                            f"(处理于 {t}, attempt={od['attempt']}->{od['attempt'] + 1})"
                        )

                        # 用当前买一生成新的挂单窗口（立即生效）
                        new_place_time = t
                        new_expire_time = t + pd.Timedelta(seconds=SELL_WAIT_SECONDS)
                        od["order_price"] = float(bid)
                        od["place_time"] = new_place_time
                        od["expire_time"] = new_expire_time
                        od["attempt"] += 1

                        # 作为“已激活订单”继续等待判定
                        still_active.append(od)
                        continue
                    else:
                        # 达到最大重挂次数 -> 失败
                        trades.append({
                            "time": expire_time,
                            "processed_at": t,
                            "action": "SELL_FAIL",
                            "qty": od["qty"],
                            "price": order_price,
                            "reason": f"timeout_after_{MAX_REPOSTS}_attempts",
                            "place_time": place_time,
                            "expire_time": expire_time,
                            "attempt": od["attempt"],
                        })
                        logger.warning(
                            f"{expire_time} | SELL 失败（重挂已达上限 {MAX_REPOSTS} 次） "
                            f"order_price={order_price} placed_at={place_time}"
                        )
                        continue

                # 3) 窗口内：(place_time, expire_time) 判定是否成交；未触发仅等待
                cond_last = (pd.notna(last) and last >= order_price)
                cond_bid_up = (pd.notna(bid) and bid > order_price)  # 严格大于，避免恒真
                if cond_last or cond_bid_up:
                    fill_price = order_price
                    sell_qty = od["qty"]
                    position -= sell_qty
                    cash += fill_price * sell_qty
                    reason = "by_lastprice_ge_order" if cond_last else "bid_breaks_above_order"

                    trades.append({
                        "time": t,
                        "action": "SELL_OK",
                        "qty": sell_qty,
                        "price": fill_price,
                        "reason": reason,
                        "place_time": place_time,
                        "expire_time": expire_time,
                        "attempt": od["attempt"],
                    })
                    logger.info(
                        f"{t} | SELL 成功 qty={sell_qty} price={fill_price:.4f} "
                        f"reason={reason} cash={cash:.2f} pos={position}"
                    )
                    # 不加入 still_active（已结束）
                else:
                    # 纯等待
                    still_active.append(od)
            # 刷新仍在等待的订单集合
            active_sells = still_active

        # B) 买入触发&成交（用 t-1 的挂单、在 t 判定）
        if i >= 1:
            prev = df.iloc[i - 1]
            t_prev = prev["TRADINGTIME"]
            ask_prev = prev["SELLPRICE01"]
            bid_prev = prev["BUYPRICE01"]

            if pd.notna(ask_prev) and pd.notna(bid_prev):
                spread_prev = ask_prev - bid_prev
                # 上一 tick 是否触发买入挂单？
                if spread_prev < SPREAD_TH and (not ONLY_WHEN_FLAT or position == 0):
                    order_price_buy = float(bid_prev)
                    # 在当前 tick 判定买入是否成交
                    cond_last_buy = (pd.notna(last) and last <= order_price_buy)
                    cond_bid_buy = (pd.notna(bid) and bid < bid_prev)  # 严格小于
                    success_buy = cond_last_buy or cond_bid_buy

                    if success_buy:
                        # 买入成交
                        fill_buy = order_price_buy
                        old_pos = position
                        position += QTY
                        cash -= fill_buy * QTY
                        avg_cost = fill_buy if old_pos == 0 else (avg_cost * old_pos + fill_buy * QTY) / (old_pos + QTY)

                        reason_buy = "by_last_vs_prev_order" if cond_last_buy else "bid_drop_vs_prev"
                        trades.append({
                            "time": t, "action": "BUY_OK", "qty": QTY,
                            "price": fill_buy, "reason": reason_buy,
                            "prev_time": t_prev, "prev_bid": bid_prev, "prev_ask": ask_prev,
                        })
                        logger.info(
                            f"{t} | BUY 成功 qty={QTY} price={fill_buy:.4f} "
                            f"reason={reason_buy} pos={position} cash={cash:.2f}"
                        )

                        # 生成卖单计划（入队；真正挂出在 placed_index 到点时）
                        sell_place_index = i + TICKS_AFTER_BUY_TO_SELL
                        if sell_place_index < len(df):
                            t_place = df.iloc[sell_place_index]["TRADINGTIME"]
                            order_price_sell = float(df.iloc[sell_place_index]["BUYPRICE01"])  # 对手价=当时买一
                            expire_time = t_place + pd.Timedelta(seconds=SELL_WAIT_SECONDS)
                            sell_queue.append({
                                "placed_index": sell_place_index,
                                "place_time": t_place,
                                "expire_time": expire_time,
                                "order_price": order_price_sell,
                                "qty": QTY
                            })
                            logger.info(
                                f"{t} | 生成 SELL 计划（将在第 {sell_place_index} 根激活） "
                                f"对手价预估={order_price_sell:.4f} 窗口至 {expire_time}"
                            )
                        else:
                            logger.warning(f"{t} | 数据不足，无法在第 {TICKS_AFTER_BUY_TO_SELL} 个 tick 放置卖单")

                    else:
                        trades.append({
                            "time": t, "action": "BUY_FAIL", "qty": QTY,
                            "price": order_price_buy, "reason": "price_conditions_not_met",
                            "prev_time": t_prev, "prev_bid": bid_prev, "prev_ask": ask_prev,
                        })
                        logger.warning(
                            f"{t} | BUY 失败（价格条件不满足） "
                            f"prev_order_price={order_price_buy}, last={last}, bid_now={bid}, prev_bid={bid_prev}"
                        )

    # 对尚未完成的激活卖单：按过期/未到期语义 Fail
    t_end = df["TRADINGTIME"].iloc[-1]
    for od in active_sells:
        place_time = od["place_time"]
        expire_time = od["expire_time"]
        reason = "eod_no_fill_within_2s" if t_end >= place_time else "eod_before_place"
        trades.append({
            "time": expire_time,
            "action": "SELL_FAIL",
            "qty": od["qty"],
            "price": od["order_price"],
            "reason": reason,
            "place_time": place_time,
            "expire_time": expire_time,
        })

    # 对仍在队列里、尚未到激活点的计划：未到放单时刻就结束 → Fail
    for plan in sell_queue:
        trades.append({
            "time": plan["place_time"],
            "action": "SELL_FAIL",
            "qty": plan["qty"],
            "price": plan["order_price"],
            "reason": "eod_before_place",
            "place_time": plan["place_time"],
            "expire_time": plan["expire_time"],
        })

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
end = time.perf_counter()
print(f"总运行时间: {end - start:.3f} 秒")