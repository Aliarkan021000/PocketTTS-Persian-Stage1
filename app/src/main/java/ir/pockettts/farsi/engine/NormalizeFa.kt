package ir.pockettts.farsi.engine

import java.math.BigInteger
import java.text.Normalizer

/**
 * پورت دقیق و ۱۰۰٪ منطبق با:
 * - engine/normalize_fa.py
 * - _transliterate_word در engine/g2p_onnx.py
 */
object NormalizeFa {

    const val ZWNJ = '\u200c'

    private val INVISIBLES = setOf(
        '\u200b', // ZWSP
        '\u200d', // ZWJ
        '\u200e', // LRM
        '\u200f', // RLM
        '\ufeff', // BOM
        '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
        '\u2066', '\u2067', '\u2068', '\u2069'
    )

    private val DIACRITICS: Set<Char> = buildSet {
        for (c in 0x064B..0x065F) add(c.toChar())
        add(0x0670.toChar())
        for (c in 0x06D6..0x06ED) add(c.toChar())
    }

    private val CHAR_MAP = mapOf(
        'ي' to "ی",
        'ى' to "ی",
        'ۍ' to "ی",
        'ې' to "ی",
        'ك' to "ک",
        'ڪ' to "ک",
        'ة' to "ه",
        'ۀ' to "ه",
        'ۂ' to "ه",
        'ۃ' to "ه",
        'أ' to "ا",
        'إ' to "ا",
        'ٱ' to "ا",
        'ٲ' to "ا",
        'ٳ' to "ا",
        'ؤ' to "و",
        'ء' to "",
        'ـ' to ""
    )

    private val PUNCT_MAP = mapOf(
        ',' to '،',
        ';' to '؛',
        '?' to '؟',
        '!' to '!',
        '．' to '.',
        '。' to '.',
        '…' to '.',
        '«' to ' ',
        '»' to ' ',
        '"' to ' ',
        '\'' to ' ',
        '“' to ' ',
        '”' to ' ',
        '’' to ' ',
        '‘' to ' ',
        '(' to ' ',
        ')' to ' ',
        '[' to ' ',
        ']' to ' ',
        '{' to ' ',
        '}' to ' ',
        '—' to ' ',
        '–' to ' ',
        '-' to ' ',
        '/' to ' ',
        '\\' to ' '
    )

    private val PERSIAN_LETTERS: Set<Char> =
        "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآ".toSet()

    private val ALLOWED: Set<Char> =
        PERSIAN_LETTERS + setOf(ZWNJ, '.', '،', '؛', '؟', '!', ':', ' ')

    private val ONES = arrayOf(
        "", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"
    )
    private val TEENS = arrayOf(
        "ده", "یازده", "دوازده", "سیزده", "چهارده",
        "پانزده", "شانزده", "هفده", "هجده", "نوزده"
    )
    private val TENS = arrayOf(
        "", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"
    )
    private val HUNDREDS = arrayOf(
        "", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"
    )
    private val SCALES = arrayOf(
        "", "هزار", "میلیون", "میلیارد", "تریلیون"
    )

    // --- جدول‌های آوانگاری کلمات لاتین از engine/g2p_onnx.py ---
    private val LATIN_DIGRAPHS = listOf(
        "sh" to "ش", "ch" to "چ", "kh" to "خ", "gh" to "غ",
        "zh" to "ژ", "ph" to "ف", "th" to "ت", "ck" to "ک",
        "qu" to "کو", "tion" to "شن", "sion" to "ژن",
        "oo" to "و", "ee" to "ی", "ea" to "ی", "ou" to "او",
        "ow" to "و", "ai" to "ای", "ay" to "ای", "oi" to "وی",
        "au" to "او", "aw" to "او"
    )

    private val LATIN_CHARS = mapOf(
        'a' to "ا", 'b' to "ب", 'c' to "ک", 'd' to "د", 'e' to "ه",
        'f' to "ف", 'g' to "گ", 'h' to "ه", 'i' to "ی", 'j' to "ج",
        'k' to "ک", 'l' to "ل", 'm' to "م", 'n' to "ن", 'o' to "و",
        'p' to "پ", 'q' to "ک", 'r' to "ر", 's' to "س", 't' to "ت",
        'u' to "و", 'v' to "و", 'w' to "و", 'x' to "کس", 'y' to "ی", 'z' to "ز"
    )

    private val LATIN_LETTER_NAMES = mapOf(
        'a' to "ای", 'b' to "بی", 'c' to "سی", 'd' to "دی", 'e' to "ئی",
        'f' to "اف", 'g' to "جی", 'h' to "اچ", 'i' to "آی", 'j' to "جی",
        'k' to "کی", 'l' to "ال", 'm' to "ام", 'n' to "ان", 'o' to "او",
        'p' to "پی", 'q' to "کیو", 'r' to "آر", 's' to "اس", 't' to "تی",
        'u' to "یو", 'v' to "وی", 'w' to "دابلیو", 'x' to "ایکس", 'y' to "وای", 'z' to "زد"
    )

    private val LATIN_EXCEPTIONS = mapOf(
        "echo" to "اکو",
        "windows" to "ویندوز",
        "python" to "پایتون"
    )

    private val LATIN_WORD_REGEX = Regex("[A-Za-z]{2,}")
    private val NUMBER_TOKEN_REGEX = Regex("(?<![A-Za-z_])(\\d+(?:[.,]\\d+)?)(?![A-Za-z_])")
    private val PERSIAN_DETECT_REGEX = Regex("[\\u0600-\\u06FF]")
    private val ASCII_ALPHA_REGEX = Regex("[A-Za-z]")

    fun transliterateLatinWords(text: String): String {
        return LATIN_WORD_REGEX.replace(text) { match ->
            val w = match.value
            val low = w.lowercase()
            LATIN_EXCEPTIONS[low]?.let { return@replace it }

            if (w.all { it.isUpperCase() } && w.length <= 5) {
                return@replace low.map { LATIN_LETTER_NAMES[it] ?: "" }.joinToString(ZWNJ.toString())
            }

            val out = StringBuilder()
            var i = 0
            while (i < low.length) {
                if (i == 0 && low[i] in "aeiou") {
                    out.append(
                        when (low[i]) {
                            'a' -> "ا"
                            'e' -> "ا"
                            'i' -> "ای"
                            'o' -> "او"
                            'u' -> "او"
                            else -> "ا"
                        }
                    )
                    i += 1
                    continue
                }
                var matched = false
                for ((dg, fa) in LATIN_DIGRAPHS) {
                    if (low.startsWith(dg, startIndex = i)) {
                        out.append(fa)
                        i += dg.length
                        matched = true
                        break
                    }
                }
                if (matched) continue
                if (i == low.length - 1 && low[i] == 'e' && low.length > 2) {
                    i += 1
                    continue
                }
                out.append(LATIN_CHARS[low[i]] ?: "")
                i += 1
            }
            var res = out.toString()
            if (res.length >= 2 && res[0] == 'س' && res[1] !in listOf('ا', 'و', 'ی')) {
                res = "ا$res"
            }
            res
        }
    }

    private fun threeDigitsToFa(n: Int): String {
        require(n in 0..999)
        val parts = mutableListOf<String>()
        val h = n / 100
        val rem = n % 100
        if (h > 0) parts.add(HUNDREDS[h])
        if (rem in 10..19) {
            parts.add(TEENS[rem - 10])
        } else {
            val t = rem / 10
            val o = rem % 10
            if (t > 0) parts.add(TENS[t])
            if (o > 0) parts.add(ONES[o])
        }
        return parts.joinToString(" و ")
    }

    fun intToFa(n: Long): String {
        if (n == 0L) return "صفر"
        if (n < 0L) return "منفی " + intToFa(-n)

        var value = n
        val groups = mutableListOf<Int>()
        while (value > 0L) {
            groups.add((value % 1000L).toInt())
            value /= 1000L
        }
        if (groups.size > SCALES.size) {
            return n.toString().map { ONES[it.digitToInt()] }.joinToString(" ")
        }
        val parts = mutableListOf<String>()
        for (i in groups.indices.reversed()) {
            val g = groups[i]
            if (g == 0) continue
            val w = threeDigitsToFa(g)
            val scale = SCALES[i]
            if (scale.isNotEmpty()) {
                parts.add(if (g == 1 && scale == "هزار") "هزار" else "$w $scale")
            } else {
                parts.add(w)
            }
        }
        return parts.joinToString(" و ")
    }

    private fun decimalToFa(intPart: String, fracPart: String): String {
        val base = intToFa(intPart.toLongOrNull() ?: 0L)
        val fracStripped = fracPart.trimEnd('0')
        if (fracStripped.isEmpty()) return base
        val fracVal = fracStripped.toLongOrNull() ?: 0L
        val fracWords = intToFa(fracVal)
        val scale = when (fracStripped.length) {
            1 -> "دهم"
            2 -> "صدم"
            3 -> "هزارم"
            else -> ""
        }
        return if (scale.isNotEmpty()) {
            "$base ممیز $fracWords $scale"
        } else {
            val digits = fracStripped.map {
                if (it == '0') "صفر" else ONES[it.digitToInt()]
            }.joinToString(" ")
            "$base ممیز $digits"
        }
    }

    fun numbersToWords(text: String): String {
        return NUMBER_TOKEN_REGEX.replace(text) { m ->
            val tok = m.groupValues[1]
            if ('.' in tok || ',' in tok) {
                val sep = if ('.' in tok) '.' else ','
                val parts = tok.split(sep, limit = 2)
                val ip = parts[0]
                val fp = parts.getOrElse(1) { "" }
                if (fp.length == 3 && ip.length <= 3 && sep == ',') {
                    val combined = (ip + fp).toLongOrNull()
                    if (combined != null) return@replace " ${intToFa(combined)} "
                }
                return@replace " ${decimalToFa(ip, fp)} "
            }
            if (tok.length > 1 && tok.startsWith('0')) {
                val words = tok.map { if (it == '0') "صفر" else ONES[it.digitToInt()] }.joinToString(" ")
                return@replace " $words "
            }
            if (tok.length > 12) {
                val words = tok.map { if (it == '0') "صفر" else ONES[it.digitToInt()] }.joinToString(" ")
                return@replace " $words "
            }
            val parsed = BigInteger(tok)
            " ${intToFa(parsed.toLong())} "
        }
    }

    fun isPhonemic(text: String): Boolean {
        return !PERSIAN_DETECT_REGEX.containsMatchIn(text) && ASCII_ALPHA_REGEX.containsMatchIn(text)
    }

    private fun foldDigits(ch: Char): Char {
        return when (ch) {
            in '۰'..'۹' -> ('0'.code + (ch.code - '۰'.code)).toChar()
            in '٠'..'٩' -> ('0'.code + (ch.code - '٠'.code)).toChar()
            else -> ch
        }
    }

    /**
     * تابع اصلی نرمال‌سازی متن فارسی قبل از ارسال به G2P
     */
    fun normalizeForModel(rawText: String, transliterateLatin: Boolean = true): String {
        if (isPhonemic(rawText)) {
            return rawText.trim()
        }
        var text = if (transliterateLatin) transliterateLatinWords(rawText) else rawText
        text = Normalizer.normalize(text, Normalizer.Form.NFC)

        // ۱. حذف کاراکترهای نامرئی و اعراب، جایگزینی حروف عربی و ارقام
        val sb = StringBuilder(text.length)
        for (ch in text) {
            if (ch in INVISIBLES || ch in DIACRITICS) continue
            val mapped = CHAR_MAP[ch]
            if (mapped != null) {
                sb.append(mapped)
            } else {
                sb.append(foldDigits(ch))
            }
        }
        text = sb.toString()

        // ۲. تبدیل علامت درصد و اعداد به معادل حروفی فارسی
        text = text.replace("٪", " درصد ").replace("%", " درصد ")
        text = numbersToWords(text)

        // ۳. یکسان‌سازی علائم نگارشی و فیلتر کاراکترهای مجاز
        val filtered = StringBuilder(text.length)
        for (ch in text) {
            val p = PUNCT_MAP[ch] ?: ch
            if (p == '
' || p == '' || p == '	') {
                filtered.append(' ')
            } else if (p in ALLOWED) {
                filtered.append(p)
            } else {
                filtered.append(' ')
            }
        }
        text = filtered.toString()

        // ۴. تمیزکاری نیم‌فاصله (ZWNJ)
        text = text.replace(Regex("$ZWNJ+"), ZWNJ.toString())
        text = text.replace(Regex("\\s*$ZWNJ\\s*"), ZWNJ.toString())
        text = text.replace(Regex("(?<![\\u0600-\\u06FF])$ZWNJ"), "")
        text = text.replace(Regex("$ZWNJ(?![\\u0600-\\u06FF])"), "")

        // ۵. تمیزکاری فاصله‌های اطراف علائم نگارشی
        text = text.replace(Regex("\\s*([.،؛؟!:])\\s*"), "$1 ")
        text = text.replace(Regex("([.،؛؟!:])(?:\\s*\\1)+"), "$1")

        // ۶. ادغام فاصله‌های تکراری و حذف علائم ابتدای خط
        text = text.replace(Regex("\\s+"), " ").trim()
        text = text.trimStart('.', '،', '؛', '؟', '!', ':', ' ')

        return text
    }
}
