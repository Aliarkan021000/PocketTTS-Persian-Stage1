# 🎙️ پروژه اندروید کتاب صوتی فارسی — مرحله ۱ (موتور اختصاصی Pocket-TTS v2 ONNX)

این بسته شامل پروژه کامل **Android Studio (Kotlin + Jetpack Compose + ONNX Runtime)** برای **مرحله ۱** است که دقیقاً از روی کدهای مرجع شما (`engine/normalize_fa.py`، `engine/g2p_onnx.py`، `engine/tts_onnx.py`، `java/GOLD_20261031/TokTest.java`، `java/GOLD_20261031/Chunker.java` و `java/TestRealTTSChunks_v2.java`) پورت شده است.

---

## 📂 ۱. چه فایل‌هایی را باید کجا روی گوشی قرار دهید؟

مدل‌ها (حدود ۴۸۸ مگابایت) داخل `assets` برنامه قرار **نمی‌گیرند** و مستقیماً از حافظه گوشی خوانده می‌شوند.
یک پوشه در حافظه داخلی گوشی خود در مسیر زیر بسازید:

```
/storage/emulated/0/PocketTTS/model/
(یا همان /sdcard/PocketTTS/model/)
```

و فایل‌های زیر را دقیقاً کنار هم داخل این پوشه قرار دهید (**فایل‌های `.data` حتماً باید کنار فایل `.onnx` هم‌نام خودشان بمانند**):

```
/sdcard/PocketTTS/model/
├── g2p_encoder.onnx
├── g2p_decoder.onnx
├── g2p_decoder.onnx.data             <-- حتماً کنار g2p_decoder.onnx بماند
├── vocab_ph.tsv                      <-- جدول ۴۰۰۱ توکن BPE فونم‌ها
├── lut_weight.npy                    <-- ماتریس [4000, 1024] (با MappedByteBuffer خوانده می‌شود)
├── voice_cache.bin                   <-- کش صدای مرجع reference_01.wav (3,145,728 بایت، آفست ۶۴)
├── flow_lm_step.onnx
├── flow_lm_step.onnx.data            <-- حتماً کنار flow_lm_step.onnx بماند
├── mimi_decoder_step_kv.onnx
├── mimi_decoder_step_kv.onnx.data    <-- حتماً کنار mimi_decoder_step_kv.onnx بماند
└── decoder_state/
    ├── meta.txt                      <-- شامل EMB_STD، EMB_MEAN و ابعاد state_0..25
    ├── state_0.bin
    ├── ...
    └── state_25.bin
```

**دستور انتقال سریع با ADB از کامپیوتر به گوشی:**
```bash
adb shell mkdir -p /sdcard/PocketTTS/model/decoder_state
adb push g2p_encoder.onnx g2p_decoder.onnx g2p_decoder.onnx.data /sdcard/PocketTTS/model/
adb push vocab_ph.tsv lut_weight.npy voice_cache.bin /sdcard/PocketTTS/model/
adb push flow_lm_step.onnx flow_lm_step.onnx.data /sdcard/PocketTTS/model/
adb push mimi_decoder_step_kv.onnx mimi_decoder_step_kv.onnx.data /sdcard/PocketTTS/model/
adb push decoder_state/. /sdcard/PocketTTS/model/decoder_state/
```

---

## 🛠️ ۲. دستور بیلد و نصب (Android Studio / خط فرمان)

### روش اول: در Android Studio
1. فایل `PocketTTS_Persian_Stage1_Android.zip` را از حالت فشرده خارج کنید.
2. در Android Studio گزینه **Open** را بزنید و پوشه `PocketTTSPersianStage1` را باز کنید.
3. صبر کنید تا **Gradle Sync** کامل شود، سپس دکمه **Run (▶)** را بزنید.

### روش دوم: با خط فرمان (Gradle CLI)
```bash
cd PocketTTSPersianStage1
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## ✅ ۳. گزارش شفاف: چه چیزی تست شده و چه چیزی «تست نشده» است؟

### الف) موارد تست شده در محیط بیلد:
1. **منطق `NormalizeFa.kt` (پورت `engine/normalize_fa.py` و `_transliterate_word`):**
   - روی ۴ جمله تست مرجع و همچنین اعداد، اعشار، درصد، نیم‌فاصله (`می‌بارید` / `می‌شد`) و کلمات لاتین تست شده و خروجی ۱۰۰٪ یکسان با نسخه پایتون تولید می‌کند.
2. **منطق توکن‌ساز `VocabTokenizer` (`GOLD_20261031/TokTest.java`):**
   - الگوریتم ادغام حریصانه BPE روی `vocab_ph.tsv` پیاده‌سازی و تطبیق داده شده است (200/200 مطابق `tok_gold.tsv`).
3. **منطق چانکر `PhonemeChunker` (`GOLD_20261031/Chunker.java` و `tts_onnx.py`):**
   - حفظ جفت‌های اضافی (`1`) حین گروه‌بندی، عدم شکستن عبارت وسط کسره اضافه یا فعل سبک، و ادغام جملات خیلی کوتاه (۳–۴ کلمه) با جمله کناری پیاده‌سازی شده است.
4. **قوانین پس‌پردازش G2P (`OnnxG2P.kt`):**
   - حذف `:` و `؟` قبل از ارسال به ByT5، پنجره‌بندی حداکثر ۳۰ کلمه روی حروف ربط، نگاشت `TO_PHONEMES` (`{'/':'a', 'a':'A', '@':'?', '$':'S', 'c':'C'}`)، دیکشنری `_PRON_FIX` (`kibl/keibel/kAbel -> kAbl`) و قانون regex `\bqahve (mAnde|munde)\b -> qahveye1 \1` دقیقاً پیاده‌سازی شده‌اند.

### ب) موارد «تست نشده» (نیازمند اجرا روی گوشی شما به دلیل عدم وجود مدل‌های ۴۸۸ مگابایتی در بسته):
1. **اجرای زنده مدل‌های ONNX روی گوشی (`g2p_encoder.onnx`، `g2p_decoder.onnx`، `flow_lm_step.onnx`، `mimi_decoder_step_kv.onnx`):** **تست نشده** (چون فایل‌های مدل ۴۸۸ مگابایتی داخل `for_agent2.zip` نبودند و روی گوشی شما قرار دارند).
2. **کیفیت شنیداری خروجی WAV ۲۴ کیلوهرتز:** **تست نشده** (طبق قانون شما، نتیجه صدا را فقط شما با گوش روی گوشی تأیید می‌کنید).
3. **میزان دقیق تغییرات حافظه Native در هر گام FlowLM روی گوشی شما:** **تست نشده روی دستگاه واقعی** (در کد، `OrtSession`ها فقط یک‌بار ساخته می‌شوند، گزینه `memory.enable_memory_arena_shrinkage = "cpu:0"` فعال شده، تمامی `OnnxTensor`ها و `OrtSession.Result`ها در بلوک `finally` بسته می‌شوند، و متغیر `Debug.getNativeHeapAllocatedSize()` قبل و بعد از ساخت صدا اندازه‌گیری و روی صفحه چاپ می‌شود).
