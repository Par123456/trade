import ccxt.async_support as ccxt_async
import ta
import schedule
import time
import os
import io
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import asyncio
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# ===================== تنظیمات اولیه =====================
CONFIG_FILE = "config.json"
LOG_FILE = "signal_logs.json"
ERROR_LOG_FILE = "error_logs.txt"
CHART_DIR = "charts"
BACKTEST_FILE = "backtest_results.json"
DEFAULT_CONFIG = {
    "telegram_token": "8145688023:AAHbPn6QgO1t7tQUnS2-kRx7FDoO0mr15tE",
    "chat_id": "6508600903",
    "symbols": ["SHIB/USDT", "BONK/USDT", "DOGE/USDT", "PEPE/USDT"],
    "timeframe": "15m",
    "higher_timeframe": "1h",  # برای تحلیل چند تایم‌فریمی
    "price_alert_percent": 5,
    "schedule_interval": 15,  # دقیقه
    "risk_reward_ratio": 2.0,  # حداقل نسبت ریسک به ریوارد
    "stop_loss_multiplier": 1.5,  # ضریب ATR برای استاپ لاس
    "take_profit_multiplier": 3.0  # ضریب ATR برای تیک پروفیت
}

# تنظیم لاگ‌گیری
logging.basicConfig(
    filename=ERROR_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# ===================== مدیریت تنظیمات =====================
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

config = load_config()

# ===================== مدیریت داده‌ها =====================
@lru_cache(maxsize=128)
async def fetch_ohlcv(exchange, symbol, timeframe, limit=200):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"خطا در دریافت داده‌های {symbol}: {str(e)}")
        return None

async def fetch_all_ohlcv(symbols, timeframe):
    async with ccxt_async.binance() as exchange:
        tasks = [fetch_ohlcv(exchange, symbol, timeframe) for symbol in symbols]
        return await asyncio.gather(*tasks, return_exceptions=True)

# ===================== محاسبه سطوح فیبوناچی =====================
def calculate_fibonacci_levels(df):
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    levels = {
        '0.0': low,
        '0.236': low + diff * 0.236,
        '0.382': low + diff * 0.382,
        '0.5': low + diff * 0.5,
        '0.618': low + diff * 0.618,
        '1.0': high
    }
    return levels

# ===================== تحلیل بک‌تستینگ =====================
def backtest_signals(df, symbol):
    signals = []
    wins = 0
    total = 0
    for i in range(2, len(df) - 1):
        if (df['rsi'].iloc[i] < 30 and df['macd'].iloc[i] > df['signal_line'].iloc[i] and 
            df['ema9'].iloc[i] > df['ema21'].iloc[i] and df['close'].iloc[i] < df['bb_lower'].iloc[i] and 
            df['volume'].iloc[i] > df['volume_ema'].iloc[i] and df['adx'].iloc[i] > 25):
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price - df['atr'].iloc[i] * config['stop_loss_multiplier']
            take_profit = entry_price + df['atr'].iloc[i] * config['take_profit_multiplier']
            for j in range(i + 1, len(df)):
                if df['low'].iloc[j] <= stop_loss:
                    total += 1
                    break
                if df['high'].iloc[j] >= take_profit:
                    wins += 1
                    total += 1
                    break
    win_rate = (wins / total * 100) if total > 0 else 0
    return {"symbol": symbol, "win_rate": win_rate, "total_trades": total}

def log_backtest_result(result):
    with open(BACKTEST_FILE, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

# ===================== لاگ‌گیری =====================
def log_signal(data):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

def log_error(msg):
    logging.error(msg)

# ===================== رسم نمودار =====================
def plot_chart(df, symbol, fib_levels):
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # نمودار قیمت
    ax1.plot(df['time'], df['close'], label='قیمت بسته شدن', color='cyan')
    ax1.plot(df['time'], df['ema9'], label='EMA9', color='yellow', linestyle='--')
    ax1.plot(df['time'], df['ema21'], label='EMA21', color='magenta', linestyle='--')
    ax1.plot(df['time'], df['vwap'], label='VWAP', color='orange', linestyle='-.')
    ax1.fill_between(df['time'], df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
    ax1.plot(df['time'], df['ichimoku_base'], label='Ichimoku Base', color='green', linestyle='--')
    for level, price in fib_levels.items():
        ax1.axhline(price, label=f'Fib {level}', linestyle=':', alpha=0.5)
    ax1.set_title(f"تحلیل {symbol}")
    ax1.set_ylabel('قیمت')
    ax1.legend()

    # نمودار RSI
    ax2.plot(df['time'], df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.legend()

    # نمودار Stochastic
    ax3.plot(df['time'], df['stoch_k'], label='%K', color='blue')
    ax3.plot(df['time'], df['stoch_d'], label='%D', color='red', linestyle='--')
    ax3.axhline(80, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(20, color='green', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Stochastic')
    ax3.legend()

    # نمودار ADX
    ax4.plot(df['time'], df['adx'], label='ADX', color='white')
    ax4.axhline(25, color='yellow', linestyle='--', alpha=0.5)
    ax4.set_ylabel('ADX')
    ax4.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ===================== گزارش PDF =====================
def make_pdf(signals):
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f"{CHART_DIR}/Signal_Report_{today}.pdf"
    os.makedirs(CHART_DIR, exist_ok=True)
    
    pdf = canvas.Canvas(filename, pagesize=A4)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.setFillColor(colors.darkblue)
    pdf.drawString(2*cm, 27*cm, f"📊 گزارش سیگنال‌های ارز دیجیتال - {today}")
    
    y = 26*cm
    pdf.setFont("Helvetica", 12)
    for sig in signals:
        text = (f"{sig['timestamp']} | {sig['symbol']} | {sig['signal']} | "
                f"قیمت: {sig['price']:.6f} | RSI: {sig['rsi']:.2f} | "
                f"ورود: {sig['entry_price']:.6f} | خروج: {sig['take_profit']:.6f} | "
                f"استاپ: {sig['stop_loss']:.6f} | ریسک/ریوارد: {sig['risk_reward']:.2f}")
        pdf.drawString(2*cm, y, text)
        y -= 0.5*cm
        if y < 2*cm:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = 27*cm
    pdf.save()
    return filename

# ===================== تحلیل ارز =====================
async def analyze(symbol, timeframe, higher_timeframe):
    try:
        # دریافت داده‌ها برای تایم‌فریم اصلی و بالاتر
        df = await fetch_ohlcv(ccxt_async.binance(), symbol, timeframe, limit=200)
        df_higher = await fetch_ohlcv(ccxt_async.binance(), symbol, higher_timeframe, limit=100)
        if df is None or df.empty or df_higher is None or df_higher.empty:
            return None

        # محاسبه اندیکاتورها
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['signal_line'] = macd.macd_signal()
        df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['volume_ema'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_cloud'] = ichimoku.ichimoku_a()

        # محاسبه فیبوناچی
        fib_levels = calculate_fibonacci_levels(df.tail(50))

        # استخراج مقادیر
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close, rsi, macd_val, sig_line = latest['close'], latest['rsi'], latest['macd'], latest['signal_line']
        ema9, ema21, bb_upper, bb_lower = latest['ema9'], latest['ema21'], latest['bb_upper'], latest['bb_lower']
        vwap, volume, volume_ema = latest['vwap'], latest['volume'], latest['volume_ema']
        stoch_k, stoch_d, atr, adx = latest['stoch_k'], latest['stoch_d'], latest['atr'], latest['adx']
        ichimoku_base, ichimoku_cloud = latest['ichimoku_base'], latest['ichimoku_cloud']

        # تحلیل تایم‌فریم بالاتر برای تأیید روند
        df_higher['ema50'] = ta.trend.EMAIndicator(df_higher['close'], window=50).ema_indicator()
        trend_confirmed = df_higher['close'].iloc[-1] > df_higher['ema50'].iloc[-1]

        # محاسبه نقاط ورود و خروج با ATR و فیبوناچی
        entry_price = close
        stop_loss = close - atr * config['stop_loss_multiplier']
        take_profit = close + atr * config['take_profit_multiplier']
        for level, price in fib_levels.items():
            if abs(close - price) / close < 0.01:  # نزدیک به سطح فیبوناچی
                if '0.382' in level or '0.5' in level:
                    entry_price = price
                if '0.618' in level:
                    take_profit = price
                if '0.236' in level:
                    stop_loss = price

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward = reward / risk if risk > 0 else 0

        # تشخیص الگوهای کندل‌استیک
        is_bullish_engulfing = (prev['close'] < prev['open'] and 
                                close > prev['open'] and 
                                close > prev['close'] and 
                                prev['open'] > prev['close'])
        is_hammer = (latest['close'] > latest['open'] and 
                     (latest['high'] - latest['close']) / (latest['close'] - latest['open']) < 0.3 and 
                     (latest['open'] - latest['low']) / (latest['close'] - latest['open']) > 2)
        is_doji = abs(latest['close'] - latest['open']) / latest['close'] < 0.001

        # منطق سیگنال با فیلترهای سخت‌گیرانه
        signal = "ℹ️ بدون سیگنال"
        warning = ""
        if (rsi < 30 and macd_val > sig_line and ema9 > ema21 and 
            close < bb_lower and volume > volume_ema * 1.5 and 
            stoch_k < 20 and stoch_k > stoch_d and adx > 25 and 
            close > ichimoku_cloud and trend_confirmed and 
            risk_reward >= config['risk_reward_ratio'] and 
            (is_bullish_engulfing or is_hammer)):
            signal = f"✅ سیگنال خرید قوی (ریسک/ریوارد: {risk_reward:.2f})"
            warning = f"⚠️ اگر قیمت به {stop_loss:.6f} رسید، معامله را ببندید (ریسک ضرر)."
        elif (rsi > 70 and macd_val < sig_line and ema9 < ema21 and 
              close > bb_upper and volume > volume_ema * 1.5 and 
              stoch_k > 80 and stoch_k < stoch_d and adx > 25 and 
              close < ichimoku_cloud and not trend_confirmed and 
              risk_reward >= config['risk_reward_ratio'] and not is_doji):
            signal = f"❌ سیگنال فروش قوی (ریسک/ریوارد: {risk_reward:.2f})"
            warning = f"⚠️ اگر قیمت به {stop_loss:.6f} رسید، معامله را ببندید (ریسک ضرر)."

        # فیلتر بازار رنج
        if atr < df['atr'].mean() * 0.5 or is_doji:
            signal = "⚠️ بازار رنج یا عدم قطعیت - بدون سیگنال"

        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'symbol': symbol,
            'signal': signal,
            'price': float(close),
            'entry_price': float(entry_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'rsi': float(rsi),
            'macd': float(macd_val),
            'ema9': float(ema9),
            'ema21': float(ema21),
            'volume': float(volume),
            'risk_reward': float(risk_reward),
            'atr': float(atr),
            'adx': float(adx)
        }

        # بک‌تستینگ
        backtest_result = await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), lambda: backtest_signals(df, symbol))
        log_backtest_result(backtest_result)

        # ارسال پیام
        msg = (f"🔍 تحلیل {symbol} ({timeframe})\n"
               f"📉 قیمت فعلی: {close:.6f}\n"
               f"🎯 نقطه ورود: {entry_price:.6f}\n"
               f"🛑 استاپ ل弄فردی لاس: {stop_loss:.6f}\n"
               f"✅ تیک پروفیت: {take_profit:.6f}\n"
               f"⚖️ ریسک/ریوارد: {risk_reward:.2f}\n"
               f"📊 RSI: {rsi:.2f} | MACD: {macd_val:.4f} | ADX: {adx:.2f}\n"
               f"📈 EMA9/21: {ema9:.4f}/{ema21:.4f}\n"
               f"📏 Bollinger: {bb_lower:.6f} - {bb_upper:.6f}\n"
               f"📊 VWAP: {vwap:.6f} | Stochastic: {stoch_k:.2f}/{stoch_d:.2f}\n"
               f"☁️ Ichimoku Cloud: {ichimoku_cloud:.6f}\n"
               f"📈 نرخ برد (بک‌تست): {backtest_result['win_rate']:.2f}% ({backtest_result['total_trades']} معامله)\n"
               f"{signal}\n{warning}")
        
        bot = Bot(token=config['telegram_token'])
        await bot.send_message(chat_id=config['chat_id'], text=msg)
        chart_img = await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), lambda: plot_chart(df, symbol, fib_levels))
        await bot.send_photo(chat_id=config['chat_id'], photo=chart_img)

        log_signal(data)
        return data

    except Exception as e:
        log_error(f"خطا در تحلیل {symbol}: {str(e)}")
        bot = Bot(token=config['telegram_token'])
        await bot.send_message(chat_id=config['chat_id'], text=f"⚠️ خطا در تحلیل {symbol}: {str(e)}")
        return None

# ===================== هشدار قیمت و سطوح کلیدی =====================
async def check_price_alerts():
    async with ccxt_async.binance() as exchange:
        for symbol in config['symbols']:
            try:
                df = await fetch_ohlcv(exchange, symbol, config['timeframe'], limit=50)
                if df is None or df.empty:
                    continue
                atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range().iloc[-1]
                ticker = await exchange.fetch_ticker(symbol)
                change_pct = ticker.get('percentage', 0)
                fib_levels = calculate_fibonacci_levels(df.tail(50))
                dynamic_threshold = max(config['price_alert_percent'], atr / df['close'].iloc[-1] * 100 * 2)
                
                # بررسی شکست سطوح فیبوناچی
                close = df['close'].iloc[-1]
                key_level_broken = any(abs(close - price) / close < 0.01 for price in fib_levels.values())
                
                if abs(change_pct) >= dynamic_threshold or key_level_broken:
                    msg = (f"⚠️ هشدار قیمت {symbol}!\n"
                           f"تغییر 24 ساعته: {change_pct:.2f}%\n"
                           f"آستانه پویا: {dynamic_threshold:.2f}% (ATR: {atr:.6f})\n")
                    if key_level_broken:
                        closest_level = min(fib_levels.items(), key=lambda x: abs(close - x[1]))
                        msg += f"🔑 شکست سطح فیبوناچی {closest_level[0]} در قیمت {closest_level[1]:.6f}"
                    bot = Bot(token=config['telegram_token'])
                    await bot.send_message(chat_id=config['chat_id'], text=msg)
            except Exception as e:
                log_error(f"خطا در هشدار قیمت {symbol}: {str(e)}")

# ===================== دستورات تلگرام =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("دریافت سیگنال", callback_data='signal')],
        [InlineKeyboardButton("گزارش PDF", callback_data='report')],
        [InlineKeyboardButton("وضعیت", callback_data='status')],
        [InlineKeyboardButton("آمار", callback_data='stats')],
        [InlineKeyboardButton("بک‌تست", callback_data='backtest')],
        [InlineKeyboardButton("سطوح فیبوناچی", callback_data='fibonacci')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("👋 ربات حرفه‌ای تحلیل ارز دیجیتال فعال شد!", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'signal':
        await query.message.reply_text("در حال تحلیل و ارسال سیگنال‌ها…")
        signals = []
        for sym in config['symbols']:
            data = await analyze(sym, config['timeframe'], config['higher_timeframe'])
            if data:
                signals.append(data)
    elif query.data == 'report':
        await query.message.reply_text("در حال ساخت گزارش PDF…")
        with open(make_pdf(signals=[]), 'rb') as f:
            await query.message.reply_document(document=f)
    elif query.data == 'status':
        msg = (f"🛠 وضعیت ربات:\n"
               f"نمادها: {', '.join(config['symbols'])}\n"
               f"بازه زمانی: {config['timeframe']} (تایم‌فریم بالاتر: {config['higher_timeframe']})\n"
               f"هشدار قیمت: {config['price_alert_percent']}%\n"
               f"ریسک/ریوارد حداقل: {config['risk_reward_ratio']}\n"
               f"استاپ لاس: {config['stop_loss_multiplier']}x ATR\n"
               f"تیک پروفیت: {config['take_profit_multiplier']}x ATR")
        await query.message.reply_text(msg)
    elif query.data == 'stats':
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                signals = [json.loads(line) for line in f]
            total = len(signals)
            buys = sum(1 for s in signals if 'خرید' in s['signal'])
            sells = sum(1 for s in signals if 'فروش' in s['signal'])
            avg_rr = sum(s['risk_reward'] for s in signals) / total if total > 0 else 0
            msg = (f"📈 آمار سیگنال‌ها:\n"
                   f"کل: {total}\nخرید: {buys}\nفروش: {sells}\n"
                   f"میانگین ریسک/ریوارد: {avg_rr:.2f}")
            await query.message.reply_text(msg)
        else:
            await query.message.reply_text("هیچ سیگنالی ثبت نشده است.")
    elif query.data == 'backtest':
        if os.path.exists(BACKTEST_FILE):
            with open(BACKTEST_FILE, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]
            msg = "📊 نتایج بک‌تست:\n"
            for res in results:
                msg += f"{res['symbol']}: نرخ برد {res['win_rate']:.2f}% ({res['total_trades']} معامله)\n"
            await query.message.reply_text(msg)
        else:
            await query.message.reply_text("هیچ بک‌تستی انجام نشده است.")
    elif query.data == 'fibonacci':
        for sym in config['symbols']:
            df = await fetch_ohlcv(ccxt_async.binance(), sym, config['timeframe'], limit=50)
            if df is not None and not df.empty:
                fib_levels = calculate_fibonacci_levels(df.tail(50))
                msg = f"📏 سطوح فیبوناچی {sym}:\n"
                for level, price in fib_levels.items():
                    msg += f"{level}: {price:.6f}\n"
                await query.message.reply_text(msg)

async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("لطفاً یک نماد وارد کنید. مثال: /add SHIB/USDT")
        return
    symbol = context.args[0].upper()
    if symbol in config['symbols']:
        await update.message.reply_text("این نماد قبلاً اضافه شده است.")
    else:
        config['symbols'].append(symbol)
        save_config(config)
        await update.message.reply_text(f"نماد {symbol} اضافه شد.")

async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("لطفاً یک نماد وارد کنید. مثال: /remove SHIB/USDT")
        return
    symbol = context.args[0].upper()
    if symbol in config['symbols']:
        config['symbols'].remove(symbol)
        save_config(config)
        await update.message.reply_text(f"نماد {symbol} حذف شد.")
    else:
        await update.message.reply_text("این نماد در لیست وجود ندارد.")

async def set_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("لطفاً بازه زمانی را وارد کنید. مثال: /settimeframe 15m")
        return
    tf = context.args[0]
    allowed = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    if tf not in allowed:
        await update.message.reply_text(f"بازه معتبر نیست. گزینه‌های مجاز: {allowed}")
        return
    config['timeframe'] = tf
    save_config(config)
    await update.message.reply_text(f"بازه زمانی به {tf} تغییر کرد.")

async def set_higher_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("لطفاً بازه زمانی بالاتر را وارد کنید. مثال: /sethighertimeframe 1h")
        return
    tf = context.args[0]
    allowed = ['15m', '30m', '1h', '4h', '1d']
    if tf not in allowed:
        await update.message.reply_text(f"بازه معتبر نیست. گزینه‌های مجاز: {allowed}")
        return
    config['higher_timeframe'] = tf
    save_config(config)
    await update.message.reply_text(f"بازه زمانی بالاتر به {tf} تغییر کرد.")

async def set_alert_percent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].replace('.', '').isdigit():
        await update.message.reply_text("لطفاً درصد معتبر وارد کنید. مثال: /setalert 5")
        return
    percent = float(context.args[0])
    config['price_alert_percent'] = percent
    save_config(config)
    await update.message.reply_text(f"هشدار قیمت به {percent}% تنظیم شد.")

async def set_risk_reward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].replace('.', '').isdigit():
        await update.message.reply_text("لطفاً نسبت ریسک/ریوارد معتبر وارد کنید. مثال: /setrr 2")
        return
    rr = float(context.args[0])
    config['risk_reward_ratio'] = rr
    save_config(config)
    await update.message.reply_text(f"نسبت ریسک/ریوارد به {rr} تنظیم شد.")

async def set_sl_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2 or not all(x.replace('.', '').isdigit() for x in context.args):
        await update.message.reply_text("لطفاً ضریب استاپ لاس و تیک پروفیت را وارد کنید. مثال: /setslttp 1.5 3.0")
        return
    sl, tp = float(context.args[0]), float(context.args[1])
    config['stop_loss_multiplier'] = sl
    config['take_profit_multiplier'] = tp
    save_config(config)
    await update.message.reply_text(f"استاپ لاس به {sl}x ATR و تیک پروفیت به {tp}x ATR تنظیم شد.")

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("دستور نامعتبر است. از منوی زیر استفاده کنید:", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("دریافت سیگنال", callback_data='signal')],
        [InlineKeyboardButton("گزارش PDF", callback_data='report')]
    ]))

# ===================== تابع اصلی =====================
async def main():
    app = Application.builder().token(config['telegram_token']).build()
    
    # ثبت دستورات
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add_symbol))
    app.add_handler(CommandHandler("remove", remove_symbol))
    app.add_handler(CommandHandler("settimeframe", set_timeframe))
    app.add_handler(CommandHandler("sethighertimeframe", set_higher_timeframe))
    app.add_handler(CommandHandler("setalert", set_alert_percent))
    app.add_handler(CommandHandler("setrr", set_risk_reward))
    app.add_handler(CommandHandler("setslttp", set_sl_tp))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(CommandHandler(None, unknown_command))

    # زمان‌بندی
    async def run_scheduled_tasks():
        while True:
            schedule.run_pending()
            await asyncio.sleep(1)

    schedule.every(config['schedule_interval']).minutes.do(lambda: asyncio.create_task(
        asyncio.gather(*[analyze(sym, config['timeframe'], config['higher_timeframe']) for sym in config['symbols']])
    ))
    schedule.every(30).minutes.do(lambda: asyncio.create_task(check_price_alerts()))
    schedule.every().day.at("23:59").do(lambda: asyncio.create_task(
        Bot(token=config['telegram_token']).send_document(
            chat_id=config['chat_id'],
            document=open(make_pdf(signals=[]), 'rb')
        )
    ))

    # اجرای ربات و زمان‌بندی
    await asyncio.gather(
        app.run_polling(),
        run_scheduled_tasks()
    )

if __name__ == "__main__":
    asyncio.run(main())
