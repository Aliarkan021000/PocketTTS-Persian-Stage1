package ir.pockettts.farsi.engine

import android.os.Debug
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.channels.FileChannel
import java.util.Random
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * موتور اصلی Pocket-TTS v2 فارسی (ONNX Runtime Android)
 * منطبق با:
 * - engine/tts_onnx.py
 * - java/TestRealTTSChunks_v2.java
 * - model_info/model_contract.json & manifest.json
 */
class PocketTtsEngine : TtsEngine {

    companion object {
        private const val TAG = "PocketTtsEngine"

        // مشخصات ثابت معماری مدل از model_contract.json و TestRealTTSChunks_v2.java
        const val SAMPLE_RATE = 24000
        const val FRAME_SAMPLES = 1920
        const val L = 6
        const val KV = 2
        const val CAP = 384
        const val H = 16
        const val D = 64
        const val LDIM = 32
        const val DIM = 1024
        const val VOICE_OFFSET = 64L
        const val VOICE_CACHE_BYTES = 3_145_728 // 6 * 2 * 64 * 16 * 64 * 4 bytes
        const val CACHE_FLOATS = L * KV * CAP * H * D // 2 * 2,359,296 / 2 = 4,718,592 floats
        const val TEMP = 0.3f

        val REQUIRED_FILES = listOf(
            "g2p_encoder.onnx",
            "g2p_decoder.onnx",
            "g2p_decoder.onnx.data",
            "vocab_ph.tsv",
            "lut_weight.npy",
            "voice_cache.bin",
            "flow_lm_step.onnx",
            "flow_lm_step.onnx.data",
            "mimi_decoder_step_kv.onnx",
            "mimi_decoder_step_kv.onnx.data"
        )
    }

    override val engineId: String = "pocket_tts_v2_fa"
    override val engineName: String = "Pocket-TTS v2 فارسی (موتور اختصاصی ONNX)"

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var g2p: OnnxG2P? = null
    private var tokenizer: VocabTokenizer? = null
    private var flowSession: OrtSession? = null
    private var mimiSession: OrtSession? = null

    private var lutBuffer: FloatBuffer? = null
    private var baseVoiceCache: FloatArray? = null
    private var embStd: FloatArray? = null
    private var embMean: FloatArray? = null
    private var mimiInitStates: Array<Any>? = null
    private var mimiStateShapes: Array<LongArray>? = null
    private var mimiStateTypes: Array<String>? = null

    private var currentModelDir: File? = null
    private var initialized = false

    override fun checkModelDirectory(modelDir: File): List<ModelFileCheck> {
        return REQUIRED_FILES.map { name ->
            val f = File(modelDir, name)
            ModelFileCheck(
                fileName = name,
                exists = f.exists() && f.isFile,
                sizeBytes = if (f.exists()) f.length() else 0L
            )
        }
    }

    @Synchronized
    override fun initialize(modelDir: File): EngineInitResult {
        val checks = checkModelDirectory(modelDir)
        if (!modelDir.exists() || !modelDir.isDirectory) {
            return EngineInitResult(
                ready = false,
                message = "پوشه مدل در مسیر ${modelDir.absolutePath} وجود ندارد.",
                modelDir = modelDir.absolutePath,
                fileChecks = checks
            )
        }

        // اگر قبلاً روی همین پوشه ساخته شده، دوباره نساز (جلوگیری از نشت حافظه Native)
        if (initialized && currentModelDir?.absolutePath == modelDir.absolutePath) {
            return EngineInitResult(
                ready = true,
                message = "موتور از قبل بارگذاری شده است (استفاده مجدد از Sessionهای موجود بدون نشت حافظه).",
                modelDir = modelDir.absolutePath,
                fileChecks = checks
            )
        }

        release()

        return try {
            // ۱. راه‌اندازی G2P و توکن‌ساز (حتی اگر فقط مدل‌های G2P برای تست فاز اول کپی شده باشند)
            g2p = OnnxG2P(env, modelDir)
            val vocabFile = File(modelDir, "vocab_ph.tsv")
            if (vocabFile.exists()) {
                tokenizer = VocabTokenizer(vocabFile)
            }

            // ۲. نگاشت حافظه lut_weight.npy با MappedByteBuffer (مصرف Heap جاوا = صفر)
            val lutFile = File(modelDir, "lut_weight.npy")
            if (lutFile.exists()) {
                lutBuffer = mapNpyFloatBuffer(lutFile)
            }

            // ۳. بارگذاری کش صدای مرجع (voice_cache.bin با ترتیب BIG_ENDIAN و آفست ۶۴)
            val vcFile = File(modelDir, "voice_cache.bin")
            if (vcFile.exists()) {
                baseVoiceCache = loadVoiceCache(vcFile)
            }

            // ۴. ساخت یک‌باره Sessionهای FlowLM و Mimi با فعال‌سازی انقباض آرنای حافظه
            val flowFile = File(modelDir, "flow_lm_step.onnx")
            val mimiFile = File(modelDir, "mimi_decoder_step_kv.onnx")
            if (flowFile.exists() && mimiFile.exists()) {
                val opts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(2)
                    setInterOpNumThreads(1)
                    addConfigEntry("memory.enable_memory_arena_shrinkage", "cpu:0")
                }
                flowSession = env.createSession(flowFile.absolutePath, opts)
                mimiSession = env.createSession(mimiFile.absolutePath, opts)
                loadMimiInitialState(modelDir)
            }

            currentModelDir = modelDir
            initialized = true
            val missing = checks.filter { !it.exists }.map { it.fileName }
            val msg = if (missing.isEmpty()) {
                "تمامی مدل‌های Pocket-TTS v2 (شامل G2P، FlowLM و Mimi) با موفقیت بارگذاری شدند."
            } else {
                "موتور G2P آماده است. فایل‌های صوتی غایب: ${missing.joinToString(", ")}"
            }
            EngineInitResult(
                ready = true,
                message = msg,
                modelDir = modelDir.absolutePath,
                fileChecks = checks
            )
        } catch (e: Exception) {
            Log.e(TAG, "خطا در راه‌اندازی موتور Pocket-TTS v2", e)
            EngineInitResult(
                ready = false,
                message = "خطا در بارگذاری مدل‌ها: ${e.message}",
                modelDir = modelDir.absolutePath,
                fileChecks = checks
            )
        }
    }

    override fun isReady(): Boolean = initialized && g2p != null

    override fun normalize(text: String): String = NormalizeFa.normalizeForModel(text)

    override fun phonemise(text: String, keepEzafe: Boolean): String {
        val activeG2p = g2p ?: throw IllegalStateException("موتور G2P هنوز مقداردهی اولیه نشده است.")
        return activeG2p.phonemise(text, normalize = true, keepEzafe = keepEzafe)
    }

    override fun runG2pAcceptanceSuite(): List<G2pTestCaseResult> {
        return TtsEngine.G2P_ACCEPTANCE_REFERENCE.map { (id, persian, expected) ->
            val t0 = System.currentTimeMillis()
            val norm = NormalizeFa.normalizeForModel(persian)
            val actual = phonemise(persian, keepEzafe = true)
            val elapsed = System.currentTimeMillis() - t0
            G2pTestCaseResult(
                id = id,
                inputPersian = persian,
                normalizedText = norm,
                expectedPhonemes = expected,
                actualPhonemes = actual,
                passed = (actual.trim() == expected.trim()),
                latencyMs = elapsed
            )
        }
    }

    override fun getMemorySnapshotMb(): Pair<Double, Double> {
        val nativeMb = Debug.getNativeHeapAllocatedSize().toDouble() / (1024.0 * 1024.0)
        val rt = Runtime.getRuntime()
        val javaMb = (rt.totalMemory() - rt.freeMemory()).toDouble() / (1024.0 * 1024.0)
        return Pair(nativeMb, javaMb)
    }

    override fun synthesizeToWav(
        text: String,
        outputWav: File,
        onStatus: ((String) -> Unit)?
    ): SynthesisResult {
        val activeG2p = g2p ?: throw IllegalStateException("G2P بارگذاری نشده است.")
        val activeTok = tokenizer ?: throw IllegalStateException("فایل vocab_ph.tsv بارگذاری نشده است.")
        val activeFlow = flowSession ?: throw IllegalStateException("فایل flow_lm_step.onnx بارگذاری نشده است.")
        val activeMimi = mimiSession ?: throw IllegalStateException("فایل mimi_decoder_step_kv.onnx بارگذاری نشده است.")
        val activeLut = lutBuffer ?: throw IllegalStateException("فایل lut_weight.npy بارگذاری نشده است.")
        val activeVoiceCache = baseVoiceCache ?: throw IllegalStateException("فایل voice_cache.bin بارگذاری نشده است.")

        val (nativeBefore, _) = getMemorySnapshotMb()
        val startNs = System.nanoTime()

        onStatus?.invoke("۱/۴ در حال نرمال‌سازی و ادغام جملات خیلی کوتاه...")
        val normalized = NormalizeFa.normalizeForModel(text)
        // ادغام جملات خیلی کوتاه (۳ تا ۴ کلمه) با جمله کناری طبق درس‌های قبلی
        val mergedSentences = PhonemeChunker.mergeShortPersianSentences(normalized, minWords = 5)

        onStatus?.invoke("۲/۴ در حال استخراج فونم با G2P (پنجره ۳۰ کلمه‌ای)...")
        val sentencePhonemes = mergedSentences.map { sent ->
            activeG2p.phonemise(sent, normalize = false, keepEzafe = true)
        }
        val fullPhonemesWithEzafe = sentencePhonemes.joinToString(" ")

        // چانکینگ فونم‌ها با حفظ جفت‌های اضافی (1) و حذف 1 در خروجی نهایی چانک
        val allChunks = mutableListOf<String>()
        for (sentPh in sentencePhonemes) {
            allChunks.addAll(PhonemeChunker.chunkPhonemes(sentPh, activeTok))
        }

        val chunkWaveforms = mutableListOf<FloatArray>()
        var totalFramesGenerated = 0
        val workCache = FloatArray(CACHE_FLOATS)
        val rng = Random(42L)

        for ((chunkIdx, chunkPh) in allChunks.withIndex()) {
            onStatus?.invoke("۳/۴ تولید صوت چانک ${chunkIdx + 1} از ${allChunks.size}: $chunkPh")
            val tokenIds = activeTok.encode(chunkPh)
            val wordCount = chunkPh.split(Regex("\\s+")).count { it.isNotBlank() }

            // ساخت تانسور تعبیه متن از LUT بدون کپی کل جدول در Heap
            val T = tokenIds.size
            val textEmb = FloatArray(T * DIM)
            for (t in 0 until T) {
                val pos = tokenIds[t] * DIM
                activeLut.position(pos)
                activeLut.get(textEmb, t * DIM, DIM)
            }

            // کپی کش صدای مرجع به کش کاری
            System.arraycopy(activeVoiceCache, 0, workCache, 0, CACHE_FLOATS)

            val maxSteps = min(max(wordCount * 8 + 20, 48), 220)
            val minGenSteps = max((wordCount * 2.2).toInt(), 3)
            val framesAfterEos = (if (wordCount <= 4) 3 else 1) + 2

            val latents = ArrayList<FloatArray>()
            var eosStep: Int? = null
            var curOffset = VOICE_OFFSET

            // گام ۰: پرامپت متنی (seq = NaN [1,1,32], text_emb = [1,T,1024])
            val nanSeq = FloatArray(LDIM) { Float.NaN }
            val step0Noise = FloatArray(LDIM) { (rng.nextGaussian() * sqrt(TEMP.toDouble())).toFloat() }

            curOffset = runFlowStep(
                session = activeFlow,
                seq = nanSeq,
                seqFrames = 1L,
                textEmb = textEmb,
                textTokens = T.toLong(),
                offset = curOffset,
                noise = step0Noise,
                cache = workCache,
                outLatent = null
            )

            // حلقه اتورگرسیو FlowLM با بستن صریح تمامی تانسورها در هر گام
            var prevLatent = nanSeq
            val emptyTextEmb = FloatArray(0)
            val curLatent = FloatArray(LDIM)

            for (step in 0 until maxSteps) {
                val noise = FloatArray(LDIM) { (rng.nextGaussian() * sqrt(TEMP.toDouble())).toFloat() }
                val eosLogit = runFlowStepAutoreg(
                    session = activeFlow,
                    seq = prevLatent,
                    emptyTextEmb = emptyTextEmb,
                    offset = curOffset,
                    noise = noise,
                    cache = workCache,
                    outLatent = curLatent
                )
                curOffset += 1L
                val frameCopy = curLatent.copyOf()
                latents.add(frameCopy)
                prevLatent = frameCopy

                if (eosLogit > 0.5f && (step + 1) >= minGenSteps && eosStep == null) {
                    eosStep = step
                }
                if (eosStep != null && step >= eosStep + framesAfterEos) {
                    break
                }
            }

            totalFramesGenerated += latents.size

            // دیکود فریم‌های نهفته با Mimi Decoder (۱۹۲۰ نمونه در هر فریم)
            val rawChunkPcm = decodeMimiFrames(activeMimi, latents)
            val trimmed = speechTrim(rawChunkPcm)
            val compressed = compressInternalPauses(trimmed)
            chunkWaveforms.add(compressed)
        }

        onStatus?.invoke("۴/۴ اتصال چانک‌ها با تطبیق بلندی صدا (Stitch) و ذخیره WAV ۲۴ کیلوهرتز...")
        val finalWaveform = stitchChunks(chunkWaveforms)
        writeWav24kMono16Bit(outputWav, finalWaveform)

        val elapsedSec = (System.nanoTime() - startNs) / 1_000_000_000.0
        val audioSec = finalWaveform.size.toDouble() / SAMPLE_RATE.toDouble()
        val rtf = if (audioSec > 0.0) elapsedSec / audioSec else 0.0
        val (nativeAfter, javaUsed) = getMemorySnapshotMb()

        return SynthesisResult(
            wavFile = outputWav,
            normalizedText = normalized,
            phonemesWithEzafe = fullPhonemesWithEzafe,
            chunksRendered = allChunks,
            totalFrames = totalFramesGenerated,
            sampleRate = SAMPLE_RATE,
            audioDurationSec = audioSec,
            synthesisTimeSec = elapsedSec,
            rtf = rtf,
            memoryReport = MemoryReport(
                nativeHeapBeforeMb = nativeBefore,
                nativeHeapAfterMb = nativeAfter,
                nativeHeapDeltaMb = nativeAfter - nativeBefore,
                javaHeapUsedMb = javaUsed
            )
        )
    }

    /**
     * اجرای گام پرامپت متنی FlowLM با بستن قطعی تمامی تانسورهای ورودی و خروجی
     */
    private fun runFlowStep(
        session: OrtSession,
        seq: FloatArray,
        seqFrames: Long,
        textEmb: FloatArray,
        textTokens: Long,
        offset: Long,
        noise: FloatArray,
        cache: FloatArray,
        outLatent: FloatArray?
    ): Long {
        val tSeq = OnnxTensor.createTensor(env, FloatBuffer.wrap(seq), longArrayOf(1L, seqFrames, LDIM.toLong()))
        val tTxt = OnnxTensor.createTensor(env, FloatBuffer.wrap(textEmb), longArrayOf(1L, textTokens, DIM.toLong()))
        val tOff = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(offset)), longArrayOf(1L))
        val tNoi = OnnxTensor.createTensor(env, FloatBuffer.wrap(noise), longArrayOf(1L, LDIM.toLong()))
        val tCac = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(cache),
            longArrayOf(L.toLong(), KV.toLong(), CAP.toLong(), H.toLong(), D.toLong())
        )
        try {
            val res = session.run(
                mapOf(
                    "sequence" to tSeq,
                    "text_embeddings" to tTxt,
                    "offset" to tOff,
                    "noise" to tNoi,
                    "cache" to tCac
                )
            )
            try {
                if (outLatent != null) {
                    val latBuf = (res.get(0) as OnnxTensor).floatBuffer
                    latBuf.get(outLatent, 0, LDIM)
                }
                val newCacheBuf = (res.get(2) as OnnxTensor).floatBuffer
                newCacheBuf.get(cache, 0, CACHE_FLOATS)
                return offset + textTokens
            } finally {
                res.close()
            }
        } finally {
            tSeq.close()
            tTxt.close()
            tOff.close()
            tNoi.close()
            tCac.close()
        }
    }

    /**
     * اجرای یک گام اتورگرسیو FlowLM و برگرداندن مقدار eos
     */
    private fun runFlowStepAutoreg(
        session: OrtSession,
        seq: FloatArray,
        emptyTextEmb: FloatArray,
        offset: Long,
        noise: FloatArray,
        cache: FloatArray,
        outLatent: FloatArray
    ): Float {
        val tSeq = OnnxTensor.createTensor(env, FloatBuffer.wrap(seq), longArrayOf(1L, 1L, LDIM.toLong()))
        val tTxt = OnnxTensor.createTensor(env, FloatBuffer.wrap(emptyTextEmb), longArrayOf(1L, 0L, DIM.toLong()))
        val tOff = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(offset)), longArrayOf(1L))
        val tNoi = OnnxTensor.createTensor(env, FloatBuffer.wrap(noise), longArrayOf(1L, LDIM.toLong()))
        val tCac = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(cache),
            longArrayOf(L.toLong(), KV.toLong(), CAP.toLong(), H.toLong(), D.toLong())
        )
        try {
            val res = session.run(
                mapOf(
                    "sequence" to tSeq,
                    "text_embeddings" to tTxt,
                    "offset" to tOff,
                    "noise" to tNoi,
                    "cache" to tCac
                )
            )
            try {
                val latBuf = (res.get(0) as OnnxTensor).floatBuffer
                latBuf.get(outLatent, 0, LDIM)
                val eosVal = (res.get(1) as OnnxTensor).floatBuffer.get(0)
                val newCacheBuf = (res.get(2) as OnnxTensor).floatBuffer
                newCacheBuf.get(cache, 0, CACHE_FLOATS)
                return eosVal
            } finally {
                res.close()
            }
        } finally {
            tSeq.close()
            tTxt.close()
            tOff.close()
            tNoi.close()
            tCac.close()
        }
    }

    private fun decodeMimiFrames(mimi: OrtSession, latents: List<FloatArray>): FloatArray {
        if (latents.isEmpty()) return FloatArray(0)
        val std = embStd ?: FloatArray(LDIM) { 1f }
        val mean = embMean ?: FloatArray(LDIM) { 0f }
        val initStates = mimiInitStates ?: throw IllegalStateException("وضعیت اولیه دیکودر Mimi بارگذاری نشده است.")
        val shapes = mimiStateShapes!!
        val types = mimiStateTypes!!

        val states = Array<Any>(26) { i ->
            when (val s = initStates[i]) {
                is FloatArray -> s.copyOf()
                is LongArray -> s.copyOf()
                is BooleanArray -> s.copyOf()
                else -> s
            }
        }

        val pcmOut = FloatArray(latents.size * FRAME_SAMPLES)
        val denorm = FloatArray(LDIM)

        for (f in latents.indices) {
            val raw = latents[f]
            for (j in 0 until LDIM) {
                denorm[j] = raw[j] * std[j] + mean[j]
            }

            val inputs = HashMap<String, OnnxTensor>(27)
            try {
                inputs["latent"] = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(denorm),
                    longArrayOf(1L, 1L, LDIM.toLong())
                )
                for (i in 0 until 26) {
                    val name = "state_$i"
                    val tensor = when (types[i]) {
                        "f32" -> OnnxTensor.createTensor(env, FloatBuffer.wrap(states[i] as FloatArray), shapes[i])
                        "i64" -> OnnxTensor.createTensor(env, LongBuffer.wrap(states[i] as LongArray), shapes[i])
                        "bool" -> OnnxTensor.createTensor(env, states[i] as BooleanArray)
                        else -> throw IllegalStateException("نوع وضعیت ناشناخته: ${types[i]}")
                    }
                    inputs[name] = tensor
                }

                val res = mimi.run(inputs)
                try {
                    val audioBuf = (res.get(0) as OnnxTensor).floatBuffer
                    audioBuf.get(pcmOut, f * FRAME_SAMPLES, FRAME_SAMPLES)
                    for (i in 0 until 26) {
                        val stTensor = res.get(i + 1) as OnnxTensor
                        when (types[i]) {
                            "f32" -> stTensor.floatBuffer.get(states[i] as FloatArray)
                            "i64" -> stTensor.longBuffer.get(states[i] as LongArray)
                            "bool" -> {
                                val bb = stTensor.byteBuffer
                                val ba = states[i] as BooleanArray
                                for (k in ba.indices) ba[k] = bb.get(k) != 0.toByte()
                            }
                        }
                    }
                } finally {
                    res.close()
                }
            } finally {
                inputs.values.forEach { it.close() }
            }
        }
        return pcmOut
    }

    /**
     * برش سکوت ابتدا و انتهای چانک (منطبق با speechTrim در TestRealTTSChunks_v2.java)
     */
    private fun speechTrim(wav: FloatArray): FloatArray {
        val win = 480
        val n = wav.size / win
        if (n < 4) return wav
        val rms = FloatArray(n)
        var peak = 0f
        for (i in 0 until n) {
            var s = 0f
            for (j in 0 until win) {
                val v = wav[i * win + j]
                s += v * v
            }
            rms[i] = sqrt((s / win).toDouble()).toFloat()
            if (rms[i] > peak) peak = rms[i]
        }
        if (peak < 1e-4f) return wav
        val th = max(peak * 0.06f, 0.004f)
        var sIdx = 0
        while (sIdx < n && rms[sIdx] <= th) sIdx++
        var eIdx = n - 1
        while (eIdx > sIdx && rms[eIdx] <= th) eIdx--
        val sSamp = max(0, (sIdx - 2) * win)
        val eSamp = min(wav.size, (eIdx + 4) * win)
        val out = wav.copyOfRange(sSamp, eSamp)
        val fadeIn = min(192, out.size)
        val fadeOut = min(360, out.size)
        for (i in 0 until fadeIn) out[i] *= i.toFloat() / fadeIn
        for (i in 0 until fadeOut) out[out.size - fadeOut + i] *= (fadeOut - 1 - i).toFloat() / fadeOut
        return out
    }

    /**
     * فشرده‌سازی مکث‌های داخلی بیش از حد بلند (_compress_pauses در engine/tts_onnx.py)
     */
    private fun compressInternalPauses(wav: FloatArray, maxMs: Int = 220, keepMs: Int = 120): FloatArray {
        val win = SAMPLE_RATE / 100 // 10ms = 240 samples
        val maxW = maxMs / 10
        val keepW = keepMs / 10
        val n = wav.size / win
        if (n <= maxW + 4) return wav
        val rms = FloatArray(n)
        var maxRms = 0f
        for (i in 0 until n) {
            var s = 0f
            for (j in 0 until win) {
                val v = wav[i * win + j]
                s += v * v
            }
            rms[i] = sqrt((s / win).toDouble()).toFloat()
            if (rms[i] > maxRms) maxRms = rms[i]
        }
        val th = max(maxRms * 0.05f, 0.003f)
        val out = ArrayList<Float>(wav.size)
        var i = 0
        while (i < n) {
            if (rms[i] < th) {
                var j = i
                while (j < n && rms[j] < th) j++
                val runLen = j - i
                val limit = if (i == 0 || j == n) maxW else keepW
                val takeW = min(runLen, limit)
                for (s in (i * win) until ((i + takeW) * win)) {
                    out.add(wav[s])
                }
                i = j
            } else {
                for (s in (i * win) until ((i + 1) * win)) {
                    out.add(wav[s])
                }
                i++
            }
        }
        return out.toFloatArray()
    }

    /**
     * اتصال چانک‌ها با نرمال‌سازی RMS میانه، فید ۸ میلی‌ثانیه‌ای و محافظ پیک 0.98 (_stitch در tts_onnx.py)
     */
    private fun stitchChunks(wavs: List<FloatArray>): FloatArray {
        val valid = wavs.filter { it.isNotEmpty() }
        if (valid.isEmpty()) return FloatArray(0)

        val rmsValues = valid.map { w ->
            var sumSq = 0.0
            var count = 0
            for (v in w) {
                if (abs(v) > 0.01f) {
                    sumSq += v * v
                    count++
                }
            }
            if (count > 100) sqrt(sumSq / count).toFloat() else 0f
        }.filter { it > 1e-4f }.sorted()

        val targetRms = if (rmsValues.isNotEmpty()) rmsValues[rmsValues.size / 2] else 0f
        val fadeSamples = (SAMPLE_RATE * 8) / 1000 // 8ms = 192 samples
        val gapSamples = (SAMPLE_RATE * 140) / 1000 // 140ms natural gap

        val totalLen = valid.sumOf { it.size } + max(0, valid.size - 1) * gapSamples
        val result = FloatArray(totalLen)
        var pos = 0

        for ((idx, rawW) in valid.withIndex()) {
            val w = rawW.copyOf()
            if (targetRms > 1e-4f) {
                var sumSq = 0.0
                var count = 0
                for (v in w) {
                    if (abs(v) > 0.01f) {
                        sumSq += v * v
                        count++
                    }
                }
                if (count > 100) {
                    val r = sqrt(sumSq / count).toFloat()
                    if (r > 1e-4f) {
                        val gain = (targetRms / r).coerceIn(0.6f, 1.6f)
                        for (k in w.indices) w[k] *= gain
                    }
                }
            }
            if (w.size > fadeSamples * 2) {
                for (k in 0 until fadeSamples) {
                    w[k] *= k.toFloat() / fadeSamples
                    w[w.size - fadeSamples + k] *= (fadeSamples - 1 - k).toFloat() / fadeSamples
                }
            }
            System.arraycopy(w, 0, result, pos, w.size)
            pos += w.size
            if (idx < valid.lastIndex) {
                pos += gapSamples
            }
        }

        var peak = 0f
        for (v in result) {
            val a = abs(v)
            if (a > peak) peak = a
        }
        if (peak > 0.98f) {
            val scale = 0.98f / peak
            for (i in result.indices) result[i] *= scale
        }
        return result
    }

    private fun mapNpyFloatBuffer(npyFile: File): FloatBuffer {
        RandomAccessFile(npyFile, "r").use { raf ->
            val hdr = ByteArray(10)
            raf.readFully(hdr)
            val headerLen = (hdr[8].toInt() and 0xFF) or ((hdr[9].toInt() and 0xFF) shl 8)
            val dataOffset = 10L + headerLen.toLong()
            val dataBytes = raf.length() - dataOffset
            return raf.channel
                .map(FileChannel.MapMode.READ_ONLY, dataOffset, dataBytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()
        }
    }

    private fun loadVoiceCache(vcFile: File): FloatArray {
        val raw = vcFile.readBytes()
        val vcFloats = FloatArray(raw.size / 4)
        ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN).asFloatBuffer().get(vcFloats)

        val baseCache = FloatArray(CACHE_FLOATS)
        val V = 64
        val inner = H * D // 1024
        for (l in 0 until L) {
            for (kv in 0 until KV) {
                val srcBase = (l * KV + kv) * V * inner
                val dstBase = (l * KV + kv) * CAP * inner
                System.arraycopy(vcFloats, srcBase, baseCache, dstBase, V * inner)
            }
        }
        return baseCache
    }

    private fun loadMimiInitialState(modelDir: File) {
        val stateDir = File(modelDir, "decoder_state")
        val metaFile = File(stateDir, "meta.txt")
        if (!metaFile.exists()) return

        val shapes = Array(26) { LongArray(0) }
        val types = Array(26) { "f32" }
        val initStates = Array<Any>(26) { FloatArray(0) }

        for (line in metaFile.readLines(Charsets.UTF_8)) {
            if (line.isBlank()) continue
            val parts = line.trim().split(" ")
            when (parts[0]) {
                "EMB_STD" -> embStd = FloatArray(LDIM) { parts[it + 1].toFloat() }
                "EMB_MEAN" -> embMean = FloatArray(LDIM) { parts[it + 1].toFloat() }
                else -> {
                    val idx = parts[0].removePrefix("state_").toInt()
                    types[idx] = parts[1]
                    val sh = if (parts.size < 3 || parts[2].isEmpty()) {
                        longArrayOf()
                    } else {
                        parts[2].split(",").map { it.toLong() }.toLongArray()
                    }
                    shapes[idx] = sh
                    val raw = File(stateDir, "state_$idx.bin").readBytes()
                    initStates[idx] = when (types[idx]) {
                        "f32" -> {
                            val fa = FloatArray(raw.size / 4)
                            ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN).asFloatBuffer().get(fa)
                            fa
                        }
                        "i64" -> {
                            val la = LongArray(raw.size / 8)
                            ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN).asLongBuffer().get(la)
                            la
                        }
                        else -> {
                            BooleanArray(raw.size) { raw[it] != 0.toByte() }
                        }
                    }
                }
            }
        }
        mimiStateShapes = shapes
        mimiStateTypes = types
        mimiInitStates = initStates
    }

    private fun writeWav24kMono16Bit(outFile: File, samples: FloatArray) {
        outFile.parentFile?.mkdirs()
        val dataBytes = samples.size * 2
        val buf = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray(Charsets.US_ASCII))
        buf.putInt(36 + dataBytes)
        buf.put("WAVEfmt ".toByteArray(Charsets.US_ASCII))
        buf.putInt(16)
        buf.putShort(1) // PCM
        buf.putShort(1) // Mono
        buf.putInt(SAMPLE_RATE)
        buf.putInt(SAMPLE_RATE * 2)
        buf.putShort(2)
        buf.putShort(16)
        buf.put("data".toByteArray(Charsets.US_ASCII))
        buf.putInt(dataBytes)
        for (v in samples) {
            val clamped = v.coerceIn(-1f, 1f)
            buf.putShort((clamped * 32767f).toInt().toShort())
        }
        FileOutputStream(outFile).use { it.write(buf.array()) }
    }

    @Synchronized
    override fun release() {
        g2p?.close()
        g2p = null
        try { flowSession?.close() } catch (_: Exception) {}
        try { mimiSession?.close() } catch (_: Exception) {}
        flowSession = null
        mimiSession = null
        initialized = false
    }
}
