# -*- coding: utf-8 -*-
import asyncio
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi import FastAPI

# ==============================================================================
# بخش ۰: راه‌اندازی وب‌سرور برای بیدار ماندن
# ==============================================================================
app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Trading bot is alive."}

# ==============================================================================
# بخش ۱: تنظیمات اصلی ربات
# ==============================================================================
load_dotenv()
API_KEY = os.getenv('COINEX_API_KEY')
SECRET_KEY = os.getenv('COINEX_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    raise ValueError("خطا: کلیدهای API یافت نشدند.")

SYMBOL_FOR_TRADING = 'BTC/USDT:USDT' 
LEVERAGE = 10; MARGIN_PER_STEP_USDT = 1.0; TAKE_PROFIT_PERCENTAGE_FROM_AVG_ENTRY = 0.01;
DCA_STEP_PERCENTAGE = 0.005; SYMBOL_FOR_DATA = "BTC/USDT"; TIMEFRAME = "15m"; DATA_LIMIT = 1000 
countbc = 3; length = 21; rsi_length = length; rsi_sell = 60.0; rsi_buy = 40.0;
macd_fast_length = 9; macd_slow_length = 26; macd_signal_length = 12; macd_threshold = 400.0;
adx_val = 20.0; adx_length = length; adx_smoothing = length;
sqz_length = 20; sqz_mult = 2.0; kc_length = 20; kc_mult = 1.5; useTrueRange = True;
sqzbuy = -700.0; sqzsell = 700.0; mtf_buy_threshold = -700.0; mtf_sell_threshold = 700.0;
fastLength_mtf = 12; slowLength_mtf = 26; signalLength_mtf = 9;

# ==============================================================================
# بخش ۲: توابع تحلیل تکنیکال (کامل)
# ==============================================================================
def rma(series: pd.Series, period: int) -> pd.Series: return series.ewm(alpha=1.0/period, adjust=False).mean()
def rsi(series: pd.Series, period: int) -> pd.Series: delta = series.diff(); up = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index); down = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index); rs = rma(up, period) / rma(down, period); return 100 - (100/(1+rs))
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series: prev_close = close.shift(1); return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
def adx_plus_minus_di(high, low, close, length_adx: int, smoothing: int): up = high.diff(); down = -low.diff(); plusDM = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index); minusDM = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index); tr_rma = rma(true_range(high, low, close), length_adx); plusDI = 100.0 * rma(plusDM, length_adx) / tr_rma; minusDI = 100.0 * rma(minusDM, length_adx) / tr_rma; dx = 100.0 * (plusDI - minusDI).abs() / (plusDI + minusDI); return plusDI, minusDI, rma(dx, smoothing)
def ema(series: pd.Series, length: int) -> pd.Series: return series.ewm(span=length, adjust=False).mean()
def sma(series: pd.Series, length: int) -> pd.Series: return series.rolling(window=length, min_periods=length).mean()
def macd_lines(close: pd.Series, fast_len: int, slow_len: int, signal_len: int): fast = ema(close, fast_len); slow = ema(close, slow_len); macd_line = fast - slow; signal_line = ema(macd_line, signal_len); return macd_line, signal_line, macd_line - signal_line
def rolling_linreg_last_y(series: pd.Series, length: int) -> pd.Series:
    x = np.arange(length); sum_x = x.sum(); sum_x2 = (x**2).sum(); denom = (length * sum_x2 - sum_x**2)
    def _calc(win: pd.Series):
        y = win.values; sum_y = y.sum(); sum_xy = (x * y).sum(); m = (length * sum_xy - sum_x * sum_y) / denom; b = (sum_y - m * sum_x) / length; return b + m * (length - 1)
    return series.rolling(window=length, min_periods=length).apply(_calc, raw=False)
def squeeze_momentum_lazybear(close, high, low, sqz_len, sqz_mult, kc_len, kc_mult, use_tr=True): basis = sma(close, sqz_len); ma = sma(close, kc_len); rng = true_range(high, low, close) if use_tr else (high - low); rangema = sma(rng, kc_len); avgValue = ((high.rolling(kc_len, min_periods=kc_len).max() + low.rolling(kc_len, min_periods=kc_len).min()) / 2.0 + sma(close, kc_len)) / 2.0; return rolling_linreg_last_y(close - avgValue, kc_len)
def compute_outHist_mtf_from_intraday(df15: pd.DataFrame) -> pd.Series: df = df15.copy(); df["day"] = df["dt"].dt.floor("D"); day_close = df.groupby("day")["close"].last(); ema_fast_day_end = day_close.ewm(span=fastLength_mtf, adjust=False).mean(); ema_slow_day_end = day_close.ewm(span=slowLength_mtf, adjust=False).mean(); macd_day_end = ema_fast_day_end - ema_slow_day_end; prev_sum_Nm1 = macd_day_end.shift(1).rolling(signalLength_mtf-1, min_periods=signalLength_mtf-1).sum(); prev_day = df["day"] - pd.Timedelta(days=1); prev_fast = prev_day.map(ema_fast_day_end); prev_slow = prev_day.map(ema_slow_day_end); alpha_fast = 2.0/(fastLength_mtf+1.0); alpha_slow = 2.0/(slowLength_mtf+1.0); ema_fast_now = alpha_fast * df["close"] + (1.0 - alpha_fast) * prev_fast; ema_slow_now = alpha_slow * df["close"] + (1.0 - alpha_slow) * prev_slow; macd_now = ema_fast_now - ema_slow_now; prev_sum_for_bar = df["day"].map(prev_sum_Nm1); signal_now = (prev_sum_for_bar + macd_now) / signalLength_mtf; return macd_now - signal_now
async def fetch_ohlcv_df(exchange_obj, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ohlcv = await exchange_obj.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit); df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"]); df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True); return df
def upsert_last_candle(df_all: pd.DataFrame, last_candle_df: pd.DataFrame) -> pd.DataFrame:
    if last_candle_df is None or len(last_candle_df) == 0: return df_all
    last_new = last_candle_df.iloc[-1]
    if len(df_all) == 0: return last_candle_df.copy()
    if int(last_new["time"]) > int(df_all.iloc[-1]["time"]): return pd.concat([df_all, last_candle_df], ignore_index=True)
    else: common_cols = df_all.columns.intersection(last_new.index); df_all.loc[df_all.index[-1], common_cols] = last_new[common_cols]; return df_all
def compute_indicators(df15: pd.DataFrame) -> pd.DataFrame: df15["rsi"] = rsi(df15["close"], rsi_length); macd_line, signal_line, _ = macd_lines(df15["close"], macd_fast_length, macd_slow_length, macd_signal_length); df15["macd_line"] = macd_line; df15["signal_line"] = signal_line; plusDI, minusDI, adx_value_series = adx_plus_minus_di(df15["high"], df15["low"], df15["close"], adx_length, adx_smoothing); df15["plusDI"] = plusDI; df15["minusDI"] = minusDI; df15["adx_value"] = adx_value_series; df15["val"] = squeeze_momentum_lazybear(df15["close"], df15["high"], df15["low"], sqz_length, sqz_mult, kc_length, kc_mult, useTrueRange); df15["outHist"] = compute_outHist_mtf_from_intraday(df15); return df15
def build_conditions(df: pd.DataFrame) -> pd.DataFrame:
    cond1_long = (df["rsi"] < rsi_buy); cond2_long = (df["macd_line"] < df["signal_line"]) & (df["macd_line"] < -macd_threshold); cond3_long = (df["plusDI"] < df["minusDI"]) & (df["adx_value"] > adx_val); cond4_long = (df["val"] < sqzbuy);
    count_long = sum(cond.astype(int) for cond in [cond1_long, cond2_long, cond3_long, cond4_long]);
    df["long_condition"] = (count_long >= countbc) & (df["outHist"] < mtf_buy_threshold);
    cond1_short = (df["rsi"] > rsi_sell); cond2_short = (df["macd_line"] > df["signal_line"]) & (df["macd_line"] > macd_threshold); cond3_short = (df["plusDI"] > df["minusDI"]) & (df["adx_value"] > adx_val); cond4_short = (df["val"] > sqzsell);
    count_short = sum(cond.astype(int) for cond in [cond1_short, cond2_short, cond3_short, cond4_short]);
    df["short_condition"] = (count_short >= countbc) & (df["outHist"] > mtf_sell_threshold);
    sig = np.where(df["long_condition"], "BUY", np.where(df["short_condition"], "SELL", None));
    df["signal"] = pd.Series(sig, index=df.index); return df

# ==============================================================================
# بخش ۳: توابع معامله‌گر و مدیریت وضعیت (کامل)
# ==============================================================================
is_position_active = False; active_position_info = {"symbol": None, "side": None}
exchange = ccxt.coinex({'apiKey': API_KEY, 'secret': SECRET_KEY, 'options': {'defaultType': 'swap'}, 'enableRateLimit': True, 'timeout': 60000})
async def get_usdt_balance():
    try: balance = await exchange.fetch_balance(); return balance['USDT']['free']
    except Exception as e: print(f"❌ خطا در دریافت موجودی: {e}"); return 0
async def get_position_info(symbol):
    try:
        positions = await exchange.fetch_positions([symbol])
        for p in positions:
            if p['symbol'] == symbol and p.get('contracts', 0) != 0: return p
        return None
    except Exception as e: print(f"❌ خطا در دریافت اطلاعات پوزیشن: {e}"); return None
def reset_state():
    global is_position_active, active_position_info
    is_position_active = False; active_position_info = {"symbol": None, "side": None}
    print("--- 🔄 وضعیت ربات ریست شد و آماده سیگنال جدید است ---")
async def close_everything(symbol):
    print("\n--- 🛑 در حال پاکسازی و بستن همه چیز ---"); side = active_position_info.get("side")
    try: await exchange.cancel_all_orders(symbol); print("✅ تمام سفارشات باز لغو شدند.")
    except Exception as e: print(f"❌ خطا در لغو سفارشات: {e}")
    if side:
        position = await get_position_info(symbol)
        if position:
            max_close_attempts = 3
            for attempt in range(max_close_attempts):
                try:
                    close_side = 'sell' if side == 'buy' else 'buy'
                    await exchange.create_market_order(symbol, close_side, abs(position['contracts']), params={'reduceOnly': True})
                    print(f"✅ پوزیشن باز با موفقیت در تلاش شماره {attempt + 1} بسته شد.")
                    break
                except Exception as e:
                    print(f"❌ تلاش شماره {attempt + 1} برای بستن پوزیشن ناموفق بود: {e}")
                    if attempt < max_close_attempts - 1: await asyncio.sleep(10)
            else: print("🔥🔥🔥 هشدار: بستن خودکار پوزیشن ناموفق بود. لطفاً به صورت دستی بررسی کنید! 🔥🔥🔥")
    reset_state()
async def monitor_position_and_tp():
    global is_position_active, active_position_info
    symbol = active_position_info["symbol"]; side = active_position_info["side"]; print(f"👁️ مانیتورینگ پوزیشن {side.upper()} شروع شد.")
    while is_position_active:
        await asyncio.sleep(30)
        try:
            position = await get_position_info(symbol)
            if not position: print("⚠️ پوزیشن دیگر وجود ندارد. در حال ریست کردن..."); await close_everything(symbol); break
            avg_entry = float(position['entryPrice']); tp_price = avg_entry * (1 + TAKE_PROFIT_PERCENTAGE_FROM_AVG_ENTRY) if side == 'buy' else avg_entry * (1 - TAKE_PROFIT_PERCENTAGE_FROM_AVG_ENTRY)
            ticker = await exchange.fetch_ticker(symbol); current_price = ticker['last']
            print(f"مانیتورینگ: میانگین ورود={avg_entry:.2f}, قیمت فعلی={current_price:.2f}, حد سود={tp_price:.2f}")
            if (side == 'buy' and current_price >= tp_price) or (side == 'sell' and current_price <= tp_price):
                print("🎉 حد سود فعال شد!"); await close_everything(symbol); break
        except Exception as e: print(f"❌ خطا در حلقه مانیتورینگ (اتصال به شبکه چک شود): {e}")
async def handle_trade_signal(symbol: str, side: str, signal_price: float):
    global is_position_active, active_position_info
    if is_position_active: print("یک پوزیشن از قبل فعال است."); return
    print(f"\n✅ اجرای سیگنال: {side.upper()} {symbol} در قیمت {signal_price:.2f}")
    try:
        await exchange.set_leverage(LEVERAGE, symbol)
        initial_value = MARGIN_PER_STEP_USDT * LEVERAGE; initial_amount = initial_value / signal_price
        print(f"در حال باز کردن پوزیشن اولیه Market..."); await exchange.create_market_order(symbol, side, initial_amount)
        position = None; max_retries = 3; retry_delay_seconds = 10
        for attempt in range(max_retries):
            print(f"تلاش شماره {attempt + 1} برای پیدا کردن پوزیشن..."); await asyncio.sleep(retry_delay_seconds)
            position = await get_position_info(symbol)
            if position: print(f"✅ پوزیشن با موفقیت پیدا شد!"); break
        if not position: raise Exception(f"پوزیشن پس از {max_retries} بار تلاش یافت نشد.")
        entry_price = float(position['entryPrice']); print(f"قیمت ورود واقعی: {entry_price:.2f}")
        is_position_active = True; active_position_info["symbol"] = symbol; active_position_info["side"] = side
        asyncio.create_task(monitor_position_and_tp())
        available_balance = await get_usdt_balance(); print(f"موجودی برای پله‌های بعدی: {available_balance:.2f} USDT")
        step = 1
        while available_balance >= MARGIN_PER_STEP_USDT:
            price_multiplier = 1 - (step * DCA_STEP_PERCENTAGE) if side == 'buy' else 1 + (step * DCA_STEP_PERCENTAGE)
            order_price = entry_price * price_multiplier; order_value = MARGIN_PER_STEP_USDT * LEVERAGE; order_amount = order_value / order_price
            print(f"در حال قرار دادن پله {step}: Limit {side.upper()} در قیمت {order_price:.2f}")
            await exchange.create_limit_order(symbol, side, order_amount, order_price)
            available_balance -= MARGIN_PER_STEP_USDT; step += 1; await asyncio.sleep(0.5)
        print(f"✅ {step-1} سفارش لیمیت پله‌ای قرار داده شد.")
    except Exception as e: print(f"❌ خطا در اجرای سیگنال: {e}"); await close_everything(symbol)

# ==============================================================================
# بخش ۴: تابع تست خودکار (جدید)
# ==============================================================================
async def run_startup_test():
    """یک پوزیشن تست باز کرده، چند دقیقه صبر کرده و سپس آن را می‌بندد."""
    print("\n" + "="*50)
    print("--- 🚦 شروع تست خودکار اتصال و معامله 🚦 ---")
    
    test_symbol = 'BTC/USDT:USDT'
    test_side = 'buy'
    test_price = 50000.0
    test_margin = 1.0
    test_leverage = 10
    wait_minutes = 3 # چند دقیقه پوزیشن تست باز بماند

    # بستن پوزیشن‌های باز قبلی
    position = await get_position_info(test_symbol)
    if position:
        print("یک پوزیشن از قبل باز است. در حال بستن آن...")
        side = 'buy' if float(position['contracts']) > 0 else 'sell'
        close_side = 'sell' if side == 'buy' else 'buy'
        await exchange.create_market_order(test_symbol, close_side, abs(position['contracts']), params={'reduceOnly': True})
        await asyncio.sleep(3)
    
    # باز کردن پوزیشن تست
    try:
        await exchange.set_leverage(test_leverage, test_symbol)
        amount = (test_margin * test_leverage) / test_price
        await exchange.create_market_order(test_symbol, test_side, amount)
        await asyncio.sleep(5)
        
        position = await get_position_info(test_symbol)
        if position:
            print(f"✅✅✅ پوزیشن تست با موفقیت باز شد! لطفاً حساب CoinEx خود را چک کنید.")
            print(f"قیمت ورود: {position['entryPrice']}")
            print(f"این پوزیشن به مدت {wait_minutes} دقیقه باز خواهد ماند و سپس به طور خودکار بسته می‌شود.")
            print("="*50)
            await asyncio.sleep(wait_minutes * 60)
            
            print(f"\n--- ⏰ پایان زمان تست. در حال بستن پوزیشن تست... ---")
            await close_everything(test_symbol)
            return True # تست موفق بود
        else:
            print("\n❌ پوزیشن تست باز نشد. لطفاً خطاها را در لاگ بررسی کنید.")
            return False # تست ناموفق بود

    except Exception as e:
        print(f"❌ خطای جدی در هنگام تست خودکار: {e}")
        return False

# ==============================================================================
# بخش ۵: حلقه اصلی ربات (کامل)
# ==============================================================================
async def trading_bot_loop():
    poll_seconds = 60; last_signal_timestamp = None
    try:
        print("\n--- 🧠 در حال آماده‌سازی داده‌ها برای استراتژی اصلی ---")
        df15 = await fetch_ohlcv_df(exchange, SYMBOL_FOR_DATA, TIMEFRAME, DATA_LIMIT); 
        print(f"✅ {len(df15)} کندل اولیه دریافت شد.")
    except Exception as e:
        print(f"❌ خطا در هنگام راه‌اندازی استراتژی: {e}"); return

    print("✅ ربات تحلیلگر و معامله‌گر آماده به کار است.")
    while True:
        try:
            if is_position_active: 
                print(f"یک پوزیشن {active_position_info.get('side', '').upper()} فعال است. منتظر بسته شدن..."); 
                await asyncio.sleep(poll_seconds)
                continue
            
            print(f"\n--- ({datetime.now().strftime('%H:%M:%S')}) در حال تحلیل بازار برای سیگنال جدید... ---")
            last_candle_df = None; max_fetch_attempts = 3
            for attempt in range(max_fetch_attempts):
                try:
                    last_candle_df = await fetch_ohlcv_df(exchange, SYMBOL_FOR_DATA, TIMEFRAME, 2)
                    if last_candle_df is not None and len(last_candle_df) == 2: break
                except Exception as e:
                    print(f"❌ تلاش شماره {attempt + 1} برای دریافت کندل ناموفق بود: {e}")
                    if attempt < max_fetch_attempts - 1: await asyncio.sleep(10)
            
            if last_candle_df is None: 
                print("🔥🔥🔥 هشدار: دریافت اطلاعات کندل جدید ناموفق بود. این چرخه تحلیل نادیده گرفته می‌شود. 🔥🔥🔥"); 
                await asyncio.sleep(poll_seconds)
                continue

            df15 = upsert_last_candle(df15, last_candle_df.iloc[[0]])
            df15 = upsert_last_candle(df15, last_candle_df.iloc[[1]])
            df15_with_signals = build_conditions(compute_indicators(df15))
            current_row = df15_with_signals.iloc[-2]
            
            if pd.notna(current_row["signal"]) and current_row['time'] != last_signal_timestamp:
                last_signal_timestamp = current_row['time']
                current_sig = str(current_row["signal"]).lower()
                signal_price = float(current_row["close"])
                print(f"🔥🔥🔥 سیگنال جدید یافت شد: {current_sig.upper()} در قیمت {signal_price:.2f} 🔥🔥🔥")
                await handle_trade_signal(symbol=SYMBOL_FOR_TRADING, side=current_sig, signal_price=signal_price)
            else: 
                print(f"قیمت فعلی: {df15.iloc[-1]['close']:.2f}. شرایط سیگنال جدید مهیا نیست.")
        except Exception as e: 
            print(f"❌ خطایی در حلقه اصلی رخ داد: {e}")
            await asyncio.sleep(poll_seconds)

# ==============================================================================
# بخش ۶: راه‌اندازی نهایی
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """ابتدا تست را اجرا کرده و سپس وارد حلقه اصلی می‌شود."""
    print("🚀 سرور وب شروع به کار کرد...")
    
    async def run_main_logic():
        test_successful = await run_startup_test()
        if test_successful:
            print("\n" + "="*50)
            print("✅ تست خودکار با موفقیت انجام شد.")
            print("🤖 در حال شروع حلقه اصلی ربات استراتژی...")
            print("="*50)
            await trading_bot_loop()
        else:
            print("\n" + "="*50)
            print("❌ تست خودکار ناموفق بود. ربات اصلی اجرا نخواهد شد.")
            print("لطفاً لاگ‌ها را برای پیدا کردن خطا بررسی کنید.")
            print("سرویس برای بررسی بیشتر فعال باقی می‌ماند.")
            print("="*50)

    asyncio.create_task(run_main_logic())
