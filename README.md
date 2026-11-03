# پروژه‌ی درس تحلیل داده — پاییز ۱۴۰۵

**مسیر:** الف — NYC TLC Trip Record Data (Yellow Taxi، ژانویه ۲۰۲۴)
**دانشجو:** آیدا عباسی — شماره دانشجویی: ۴۰۱۱۷۰۸۹۷
**وضعیت فعلی:** فاز ۱ تکمیل‌شده (برچسب `phase-1`)

سؤال تحلیلی اصلی پروژه و جزئیات کامل در [`PROJECT_CARD.md`](PROJECT_CARD.md)
آمده است. خلاصه: آیا تقاضای ساعتی سوارشدن تاکسی زرد نیویورک را می‌توان از
روی ساعت روز، روز هفته و تاریخچه‌ی اخیر پیش‌بینی کرد، به‌گونه‌ای که برای
برنامه‌ریزی توزیع ناوگان مفید باشد؛ با استفاده از داده‌ی رسمی و عمومی NYC
TLC برای ژانویه‌ی ۲۰۲۴.

## ساختار مخزن

```
README.md               این فایل
PROJECT_CARD.md          کارت پروژه (سؤال، واحد تحلیل، خط مبنا، محدودیت‌ها)
data/
  README.md              منبع داده، لینک دانلود، تاریخ دریافت، تاریخچه‌ی بازسازی
  raw/                    فایل‌های خام parquet/csv (در .gitignore — در گیت‌هاب نیست)
  processed/              خروجی‌های پردازش‌شده‌ی کوچک (قابل‌کامیت)
src/
  analysis.py             منطق مشترک بارگذاری/ممیزی/پاک‌سازی/تحلیل (منبع واحد حقیقت)
  build_notebook.py        نوت‌بوک را از نو می‌سازد
  build_report.py          reports/phase1.pdf را از نو می‌سازد
notebooks/
  phase1_exploration.ipynb   نوت‌بوک اجراشده با خروجی کامل
figures/                  نمودارهای PNG خروجی تحلیل
reports/
  phase1.pdf               گزارش نهایی فاز ۱ (خروجی رسمی)
  phase2/ phase3/ phase4/ final/   برای فازهای بعدی
configs/                  برای تنظیمات فازهای بعدی (در فاز ۱ خالی)
models/                   برای مدل‌های فازهای بعدی (در فاز ۱ خالی)
requirements.txt
```

## بازتولید کامل تحلیل

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ۱. دانلود داده‌ی خام (نگاه کنید به data/README.md برای جزئیات/لینک‌ها)
mkdir -p data/raw && cd data/raw
curl -sS -o yellow_tripdata_2024-01.parquet \
  "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
curl -sS -o taxi_zone_lookup.csv \
  "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
cd ../..

# ۲. اجرای مستقیم خط‌لوله‌ی تحلیل (تولید جداول پردازش‌شده و نمودارها)
python3 src/analysis.py

# ۳. ساخت و اجرای نوت‌بوک
python3 src/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/phase1_exploration.ipynb

# ۴. ساخت گزارش PDF نهایی
python3 src/build_report.py
```

## فازهای پروژه

| فاز | موضوع | وضعیت |
|---|---|---|
| ۱ | فهم مسئله، شناخت داده، کیفیت و تحلیل اکتشافی | ✅ تکمیل (`phase-1`) |
| ۲ | آماده‌سازی داده، مهندسی ویژگی و کشف الگو | در انتظار |
| ۳ | مدل‌سازی پیش‌بینانه و طراحی آزمایش | در انتظار |
| ۴ | تفسیرپذیری، تحلیل توسعه‌یافته و تصمیم‌گیری | در انتظار |
| ۵ | استحکام، مطالعه‌ی موردی واقعی و نسخه‌ی نهایی | در انتظار |
