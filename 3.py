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
DEFAULT_CONFIG = {
    "telegram_token": "YOUR_TELEGRAM_TOKEN",
    "chat_id": "YOUR_CHAT_ID",
    "symbols": ["SHIB/USDT", "BONK/USDT", "DOGE/USDT", "PEPE/USDT"],
    "timeframe": "15m",
    "price_alert_percent": 5,
    "schedule_interval": 15  # دقیقه
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
async def fetch_ohlcv(exchange, symbol, timeframe, limit=100):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        logging.error(f"خطا در دریافت داده‌های {symbol}: {str(e)}")
        return None

async def fetch_all_ohlcv(symbols, timeframe):
    async with ccxt_async.binance() as exchange:
        tasks = [fetch_ohlcv(exchange, symbol, timeframe) for symbol in symbols]
        return await asyncio.gather(*tasks, return_exceptions=True)

# ===================== لاگ‌گیری =====================
def log_signal(data):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

def log_error(msg):
    logging.error(msg)

# ===================== رسم نمودار =====================
def plot_chart(df, symbol):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # نمودار قیمت
    ax1.plot(df['time'], df['close'], label='قیمت بسته شدن', color='cyan')
    ax1.plot(df['time'], df['ema9'], label='EMA9', color='yellow', linestyle='--')
    ax1.plot(df['time'], df['ema21'], label='EMA21', color='magenta', linestyle='--')
    ax1.fill_between(df['time'], df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
    ax1.set_title(f"تحلیل {symbol}")
    ax1.set_ylabel('قیمت')
    ax1.legend()

    # نمودار RSI
    ax2.plot(df['time'], df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.legend()

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
                f"قیمت: {sig['price']:.6f} | RSI: {sig['rsi']:.2f}")
        pdf.drawString(2*cm, y, text)
        y -= 0.5*cm
        if y < 2*cm:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = 27*cm
    pdf.save()
    return filename

# ===================== تحلیل ارز =====================
async def analyze(symbol, timeframe):
    try:
        df = await fetch_ohlcv(ccxt_async.binance(), symbol, timeframe)
        if df is None or df.empty:
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
        df['volume_ema'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()

        # استخراج مقادیر
        latest = df.iloc[-1]
        close, rsi, macd_val, sig_line = latest['close'], latest['rsi'], latest['macd'], latest['signal_line']
        ema9, ema21, bb_upper, bb_lower = latest['ema9'], latest['ema21'], latest['bb_upper'], latest['bb_lower']
        volume, volume_ema = latest['volume'], latest['volume_ema']

        # منطق سیگنال
        signal = "ℹ️ بدون سیگنال"
        if rsi < 30 and macd_val > sig_line and ema9 > ema21 and close < bb_lower and volume > volume_ema:
            signal = "✅ سیگنال خرید قوی"
        elif rsi > 70 and macd_val < sig_line and ema9 < ema21 and close > bb_upper and volume > volume_ema:
            signal = "❌ سیگنال فروش قوی"

        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'symbol': symbol,
            'signal': signal,
            'price': float(close),
            'rsi': float(rsi),
            'macd': float(macd_val),
            'ema9': float(ema9),
            'ema21': float(ema21),
            'volume': float(volume)
        }

        log_signal(data)
        msg = (f"🔍 تحلیل {symbol}\n"
               f"📉 قیمت: {close:.6f}\n"
               f"📊 RSI: {rsi:.2f} | MACD: {macd_val:.4f}\n"
               f"📈 EMA9/21: {ema9:.4f}/{ema21:.4f}\n"
               f"📏 Bollinger: {bb_lower:.6f} - {bb_upper:.6f}\n"
               f"{signal}")
        
        bot = Bot(token=config['telegram_token'])
        await bot.send_message(chat_id=config['chat_id'], text=msg)
        chart_img = await asyncio.get_event_loop().run_in_executor(ThreadPoolExecutor(), lambda: plot_chart(df, symbol))
        await bot.send_photo(chat_id=config['chat_id'], photo=chart_img)

        return data

    except Exception as e:
        log_error(f"خطا در تحلیل {symbol}: {str(e)}")
        bot = Bot(token=config['telegram_token'])
        await bot.send_message(chat_id=config['chat_id'], text=f"⚠️ خطا در تحلیل {symbol}: {str(e)}")
        return None

# ===================== هشدار قیمت =====================
async def check_price_alerts():
    async with ccxt_async.binance() as exchange:
        for symbol in config['symbols']:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                change_pct = ticker.get('percentage', 0)
                if abs(change_pct) >= config['price_alert_percent']:
                    msg = f"⚠️ هشدار قیمت {symbol}!\nتغییر 24 ساعته: {change_pct:.2f}%"
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
        [InlineKeyboardButton("آمار", callback_data='stats')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("👋 ربات تحلیل ارز دیجیتال فعال شد!", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'signal':
        await query.message.reply_text("در حال تحلیل و ارسال سیگنال‌ها…")
        signals = []
        for sym in config['symbols']:
            data = await analyze(sym, config['timeframe'])
            if data:
                signals.append(data)
    elif query.data == 'report':
        await query.message.reply_text("در حال ساخت گزارش PDF…")
        with open(make_pdf(signals=[]), 'rb') as f:
            await query.message.reply_document(document=f)
    elif query.data == 'status':
        msg = (f"🛠 وضعیت ربات:\n"
               f"نمادها: {', '.join(config['symbols'])}\n"
               f"بازه زمانی: {config['timeframe']}\n"
               f"هشدار قیمت: {config['price_alert_percent']}%")
        await query.message.reply_text(msg)
    elif query.data == 'stats':
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                signals = [json.loads(line) for line in f]
            total = len(signals)
            buys = sum(1 for s in signals if 'خرید' in s['signal'])
            sells = sum(1 for s in signals if 'فروش' in s['signal'])
            msg = f"📈 آمار سیگنال‌ها:\nکل: {total}\nخرید: {buys}\nفروش: {sells}"
            await query.message.reply_text(msg)
        else:
            await query.message.reply_text("هیچ سیگنالی ثبت نشده است.")

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

async def set_alert_percent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].replace('.', '').isdigit():
        await update.message.reply_text("لطفاً درصد معتبر وارد کنید. مثال: /setalert 5")
        return
    percent = float(context.args[0])
    config['price_alert_percent'] = percent
    save_config(config)
    await update.message.reply_text(f"هشدار قیمت به {percent}% تنظیم شد.")

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
    app.add_handler(CommandHandler("setalert", set_alert_percent))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(CommandHandler(None, unknown_command))

    # زمان‌بندی
    async def run_scheduled_tasks():
        while True:
            schedule.run_pending()
            await asyncio.sleep(1)

    schedule.every(config['schedule_interval']).minutes.do(lambda: asyncio.create_task(
        asyncio.gather(*[analyze(sym, config['timeframe']) for sym in config['symbols']])
    ))
    schedule.every(1).hour.do(lambda: asyncio.create_task(check_price_alerts()))
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
