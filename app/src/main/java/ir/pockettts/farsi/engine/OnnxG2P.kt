package ir.pockettts.farsi.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.LongBuffer

/**
 * پورت دقیق engine/g2p_onnx.py
 * مسیر تبدیل:
 * متن فارسی -> NormalizeFa -> برش پنجره‌های حداکثر ۳۰ کلمه‌ای روی حروف ربط ->
 * انکودر/دیکودر ByT5 ONNX (آفست بایت + ۳) -> نگاشت TO_PHONEMES ->
 * اصلاح تلفظ‌های _PRON_FIX و «قهوه مانده»
 */
class OnnxG2P(
    private val env: OrtEnvironment,
    modelDir: File
) {
    companion object {
        private const val PAD = 0L
        private const val EOS = 1L
        private const val OFFSET = 3
        private const val G2P_WINDOW_WORDS = 30

        private val G2P_CUT_WORDS = setOf(
            "و", "یا", "ولی", "اما", "که", "زیرا", "چون",
            "پس", "سپس", "بنابراین", "همچنین"
        )

        private val TO_PHONEMES = mapOf(
            '/' to 'a',
            'a' to 'A',
            '@' to '?',
            '$' to 'S',
            'c' to 'C'
        )

        private val PRON_FIX = mapOf(
            "bo?d" to "ba?d",
            "kibl" to "kAbl",
            "kible1" to "kAble1",
            "keibel" to "kAbl",
            "keibele1" to "kAble1",
            "kAbel" to "kAbl",
            "kAbele1" to "kAble1"
        )

        private val QAHVE_MANDE_REGEX = Regex("\\bqahve (mAnde|munde)\\b")
        private val SENTENCE_SPLIT_REGEX = Regex("(?<=[.؟!؛])\\s+")
    }

    private val encSession: OrtSession
    private val decSession: OrtSession

    init {
        val encFile = File(modelDir, "g2p_encoder.onnx")
        val decFile = File(modelDir, "g2p_decoder.onnx")
        require(encFile.exists()) { "فایل g2p_encoder.onnx در مسیر ${encFile.absolutePath} یافت نشد." }
        require(decFile.exists()) { "فایل g2p_decoder.onnx در مسیر ${decFile.absolutePath} یافت نشد." }

        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            addConfigEntry("memory.enable_memory_arena_shrinkage", "cpu:0")
        }
        encSession = env.createSession(encFile.absolutePath, opts)
        decSession = env.createSession(decFile.absolutePath, opts)
    }

    /**
     * اجرای استنتاج ByT5 روی یک قطعه کوتاه (حداکثر ۳۰ کلمه)
     */
    private fun runWindow(windowText: String, maxLen: Int = 512): String {
        val cleanWindow = windowText.trim()
        if (cleanWindow.isEmpty()) return ""

        val utf8Bytes = cleanWindow.toByteArray(Charsets.UTF_8)
        val inputIds = LongArray(utf8Bytes.size + 1)
        for (i in utf8Bytes.indices) {
            inputIds[i] = (utf8Bytes[i].toInt() and 0xFF).toLong() + OFFSET
        }
        inputIds[utf8Bytes.size] = EOS

        val seqLen = inputIds.size.toLong()
        val decodedIds = mutableListOf(PAD)

        val inputTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(inputIds),
            longArrayOf(1L, seqLen)
        )

        try {
            val encResult = encSession.run(mapOf("input_ids" to inputTensor))
            try {
                val hiddenTensor = encResult.get(0) as OnnxTensor
                for (step in 0 until maxLen) {
                    val decArr = decodedIds.toLongArray()
                    val decTensor = OnnxTensor.createTensor(
                        env,
                        LongBuffer.wrap(decArr),
                        longArrayOf(1L, decArr.size.toLong())
                    )
                    try {
                        val decResult = decSession.run(
                            mapOf(
                                "decoder_input_ids" to decTensor,
                                "encoder_hidden" to hiddenTensor
                            )
                        )
                        try {
                            @Suppress("UNCHECKED_CAST")
                            val logits = decResult.get(0).value as Array<Array<FloatArray>>
                            val lastStepLogits = logits[0][logits[0].size - 1]
                            var bestIdx = 0
                            var bestVal = Float.NEGATIVE_INFINITY
                            for (v in lastStepLogits.indices) {
                                if (lastStepLogits[v] > bestVal) {
                                    bestVal = lastStepLogits[v]
                                    bestIdx = v
                                }
                            }
                            if (bestIdx.toLong() == EOS) break
                            decodedIds.add(bestIdx.toLong())
                        } finally {
                            decResult.close()
                        }
                    } finally {
                        decTensor.close()
                    }
                }
            } finally {
                encResult.close()
            }
        } finally {
            inputTensor.close()
        }

        val outBytes = ArrayList<Byte>(decodedIds.size)
        for (id in decodedIds) {
            if (id in OFFSET.toLong() until (256 + OFFSET).toLong()) {
                outBytes.add((id - OFFSET).toByte())
            }
        }
        val rawStr = ByteArray(outBytes.size) { outBytes[it] }.toString(Charsets.UTF_8)
        val sb = StringBuilder(rawStr.length)
        for (ch in rawStr) {
            sb.append(TO_PHONEMES[ch] ?: ch)
        }
        return sb.toString()
    }

    /**
     * تقسیم جملات طولانی‌تر از ۳۰ کلمه روی حروف ربط یا ویرگول (منطبق با _split_for_g2p)
     */
    fun splitForG2P(text: String): List<String> {
        val words = text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.size <= G2P_WINDOW_WORDS) return listOf(text)

        val chunks = mutableListOf<String>()
        var remaining = words
        while (remaining.size > G2P_WINDOW_WORDS) {
            var cut: Int? = null
            val upper = minOf(remaining.size - 4, 27)
            if (upper > 15) {
                for (i in (upper - 1) downTo 15) {
                    val prevHasComma = remaining[i - 1].endsWith("،")
                    val isCutWord = remaining[i].trimEnd('،') in G2P_CUT_WORDS
                    if (prevHasComma || isCutWord) {
                        cut = i
                        break
                    }
                }
            }
            val finalCut = cut ?: 24
            chunks.add(remaining.subList(0, finalCut).joinToString(" "))
            remaining = remaining.subList(finalCut, remaining.size)
        }
        if (remaining.isNotEmpty()) {
            chunks.add(remaining.joinToString(" "))
        }
        return chunks
    }

    /**
     * تبدیل متن فارسی به فونم با اعمال تمام قوانین اصلاح تلفظ (_PRON_FIX و qahveye1 mAnde)
     */
    fun phonemise(
        rawText: String,
        normalize: Boolean = true,
        keepEzafe: Boolean = true
    ): String {
        var text = NormalizeFa.transliterateLatinWords(rawText)
        if (normalize) {
            text = NormalizeFa.normalizeForModel(text, transliterateLatin = false)
        }
        if (text.isBlank()) return ""

        val sentences = text.split(SENTENCE_SPLIT_REGEX)
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        val parts = mutableListOf<String>()
        for (sent in sentences) {
            // حذف ؟ و : قبل از G2P (طبق خط ۱۵۴ فایل engine/g2p_onnx.py: دونقطه انتهای جمله باعث تکرار کلمه آخر در ByT5 می‌شود)
            val cleanedSent = sent
                .replace("؟", "")
                .replace("?", "")
                .replace(":", "")
                .trim()
            if (cleanedSent.isEmpty()) continue

            for (sub in splitForG2P(cleanedSent)) {
                val ph = runWindow(sub)
                if (ph.isNotBlank()) {
                    parts.add(ph.trim())
                }
            }
        }

        var out = parts.joinToString(" ")
        if (PRON_FIX.isNotEmpty()) {
            out = out.split(" ").joinToString(" ") { w -> PRON_FIX[w] ?: w }
        }
        out = QAHVE_MANDE_REGEX.replace(out, "qahveye1 $1")

        return if (keepEzafe) {
            out
        } else {
            out.replace("1", "")
        }
    }

    fun close() {
        try { encSession.close() } catch (_: Exception) {}
        try { decSession.close() } catch (_: Exception) {}
    }
}
