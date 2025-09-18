# -*- coding: utf-8 -*-
import asyncio
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi import FastAPI
import httpx

# ==============================================================================
# بخش ۰: راه‌اندازی وب‌سرور و متغیرهای سراسری
# ==============================================================================
app = FastAPI(); bot_task = None
@app.get("/")
async def health_check():
    if bot_task and not bot_task.done(): return {"status": "ok", "message": "Trading bot is running."}
    else: return {"status": "error", "message": "Trading bot task is not running or has finished."}

# ==============================================================================
# بخش ۱: تنظیمات اصلی ربات
# ==============================================================================
load_dotenv(); API_KEY = os.getenv('COINEX_API_KEY'); SECRET_KEY = os.getenv('COINEX_SECRET_KEY')
if not API_KEY or not SECRET_KEY: raise ValueError("خطا: کلیدهای API یافت نشدند.")

# --- تنظیمات زمان‌بندی ---
SIGNAL_CHECK_INTERVAL_SECONDS = 300      # 5 دقیقه
TP_MONITOR_INTERVAL_SECONDS = 20         # 20 ثانیه
CLOSURE_MONITOR_INTERVAL_SECONDS = 300   # 5 دقیقه
SELF_PING_INTERVAL_SECONDS = 240         # 4 دقیقه برای بیدار ماندن (کمی کمتر از 5 دقیقه UptimeRobot)

# --- تنظیمات استراتژی ---
SYMBOL_FOR_TRADING = 'BTC/USDT:USDT'; LEVERAGE = 10; MARGIN_PER_STEP_USDT = 1.0;
TAKE_PROFIT_PERCENTAGE_FROM_AVG_ENTRY = 0.01; DCA_STEP_PERCENTAGE = 0.005;
TAKE_PROFIT_1_PERCENTAGE = 0.005; TAKE_PROFIT_2_PERCENTAGE = 0.01; CLOSE_RATIO_TP1 = 0.5;
SYMBOL_FOR_DATA = "BTC/USDT"; TIMEFRAME = "15m"; DATA_LIMIT = 1000 
countbc = 3; length = 21; rsi_length = length; rsi_sell = 60.0; rsi_buy = 40.0;
macd_fast_length = 9; macd_slow_length = 26; macd_signal_length = 12; macd_threshold = 400.0;
adx_val = 20.0; adx_length = length; adx_smoothing = length;
sqz_length = 20; sqz_mult = 2.0; kc_length = 20; kc_mult = 1.5; useTrueRange = True;
sqzbuy = -700.0; sqzsell = 700.0; mtf_buy_threshold = -700.0; mtf_sell_threshold = 700.0;
fastLength_mtf = 12; slowLength_mtf = 26; signalLength_mtf = 9;

# ==============================================================================
# بخش ۲: توابع تحلیل تکنیکال (کامل و بدون تغییر)
# ==============================================================================
# ... (تمام توابع تحلیلی شما در اینجا قرار دارند - برای کوتاهی حذف شده)
def rma(series: pd.Series, period: int) -> pd.Series: return series.ewm(alpha=1.0/period, adjust=False).mean()
# ... و بقیه ...

# ==============================================================================
# بخش ۳: توابع معامله‌گر و مدیریت وضعیت
# ==============================================================================
is_position_active = False; active_position_info = {"symbol": None, "side": None, "stage": 1}
exchange = ccxt.coinex({'apiKey': API_KEY, 'secret': SECRET_KEY, 'options': {'defaultType': 'swap'}, 'enableRateLimit': True, 'timeout': 60000})
# ... (تمام توابع معامله‌گر شما در اینجا قرار دارند - برای کوتاهی حذف شده)
async def get_position_info(symbol):
    # ...
def reset_state():
    # ...
async def close_everything(symbol):
    # ...
async def monitor_position_and_tp():
    # ...
async def handle_trade_signal(symbol: str, side: str, signal_price: float):
    # ...

# ==============================================================================
# بخش ۴: حلقه اصلی ربات (اصلاح شده)
# ==============================================================================
async def trading_bot_loop():
    print("--- بررسی و پاکسازی اولیه ---")
    await exchange.cancel_all_orders(SYMBOL_FOR_TRADING)
    position = await get_position_info(SYMBOL_FOR_TRADING)
    if position:
        print("یک پوزیشن از قبل باز است. لطفاً به صورت دستی آن را مدیریت کنید.")
        return # در صورت وجود پوزیشن، حلقه شروع نمی‌شود
        
    last_signal_timestamp = None
    try:
        print("\n--- 🧠 در حال آماده‌سازی داده‌ها برای استراتژی اصلی ---")
        df15 = await fetch_ohlcv_df(exchange, SYMBOL_FOR_DATA, TIMEFRAME, DATA_LIMIT); 
        print(f"✅ {len(df15)} کندل اولیه دریافت شد.")
    except Exception as e: print(f"❌ خطا در هنگام راه‌اندازی استراتژی: {e}"); return

    print("✅ ربات تحلیلگر و معامله‌گر آماده به کار است.")
    while True:
        try:
            if is_position_active: 
                print(f"یک پوزیشن {active_position_info.get('side', '').upper()} فعال است. منتظر بسته شدن...")
            else:
                # ... (منطق اصلی تحلیل سیگنال شما بدون تغییر) ...
                print(f"\n--- ({datetime.now().strftime('%H:%M:%S')}) در حال تحلیل بازار برای سیگنال جدید... ---")
                # ...
        except Exception as e: 
            print(f"❌ خطایی در حلقه اصلی رخ داد: {e}")
        
        await asyncio.sleep(SIGNAL_CHECK_INTERVAL_SECONDS)

# ==============================================================================
# بخش ۵: راه‌اندازی نهایی
# ==============================================================================
async def self_ping_loop():
    """هر ۴ دقیقه یک بار به خودش پینگ می‌زند تا بیدار بماند."""
    await asyncio.sleep(10)
    render_url = os.getenv('RENDER_EXTERNAL_URL')
    if not render_url: print("⚠️ هشدار: آدرس خارجی Render یافت نشد. قابلیت self-ping غیرفعال است."); return
    print(f"✅ قابلیت بیدار نگه داشتن خودکار روی آدرس {render_url} فعال شد.")
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(render_url)
        except Exception: pass
        await asyncio.sleep(SELF_PING_INTERVAL_SECONDS)

@app.on_event("startup")
async def startup_event():
    global bot_task
    print("🚀 سرور وب شروع به کار کرد. در حال فعال کردن منطق اصلی ربات و حلقه پینگ...")
    bot_task = asyncio.create_task(trading_bot_loop())
    asyncio.create_task(self_ping_loop())

@app.on_event("shutdown")
async def shutdown_event():
    if bot_task:
        bot_task.cancel(); print("🛑 تسک ربات لغو شد.")
