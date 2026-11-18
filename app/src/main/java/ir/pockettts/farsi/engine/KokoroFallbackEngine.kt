package ir.pockettts.farsi.engine

import android.os.Debug
import java.io.File

/**
 * گزینه دوم اختیاری پشت رابط TtsEngine
 * توجه: موتور پیش‌فرض برنامه PocketTtsEngine (Pocket-TTS v2 فارسی) است.
 */
class KokoroFallbackEngine : TtsEngine {
    override val engineId: String = "kokoro_secondary"
    override val engineName: String = "Kokoro / Ava-82M (موتور دوم - غیرپیش‌فرض)"

    private var ready = false

    override fun checkModelDirectory(modelDir: File): List<ModelFileCheck> {
        val f = File(modelDir, "ava_kokoro_fp16.onnx")
        return listOf(ModelFileCheck("ava_kokoro_fp16.onnx", f.exists(), if (f.exists()) f.length() else 0L))
    }

    override fun initialize(modelDir: File): EngineInitResult {
        val checks = checkModelDirectory(modelDir)
        ready = checks.all { it.exists }
        return EngineInitResult(
            ready = ready,
            message = if (ready) "موتور دوم Kokoro آماده است." else "فایل ava_kokoro_fp16.onnx یافت نشد. لطفاً از موتور پیش‌فرض Pocket-TTS v2 استفاده کنید.",
            modelDir = modelDir.absolutePath,
            fileChecks = checks
        )
    }

    override fun isReady(): Boolean = ready

    override fun normalize(text: String): String = NormalizeFa.normalizeForModel(text)

    override fun phonemise(text: String, keepEzafe: Boolean): String {
        return NormalizeFa.normalizeForModel(text)
    }

    override fun runG2pAcceptanceSuite(): List<G2pTestCaseResult> {
        return TtsEngine.G2P_ACCEPTANCE_REFERENCE.map { (id, persian, expected) ->
            val norm = normalize(persian)
            G2pTestCaseResult(
                id = id,
                inputPersian = persian,
                normalizedText = norm,
                expectedPhonemes = expected,
                actualPhonemes = "(موتور Kokoro فاقد G2P ByT5 اختصاصی شماست — برای تست قبولی G2P موتور پیش‌فرض Pocket-TTS v2 را انتخاب کنید)",
                passed = false,
                latencyMs = 0L
            )
        }
    }

    override fun synthesizeToWav(
        text: String,
        outputWav: File,
        onStatus: ((String) -> Unit)?
    ): SynthesisResult {
        throw IllegalStateException("لطفاً از موتور پیش‌فرض Pocket-TTS v2 فارسی استفاده نمایید.")
    }

    override fun getMemorySnapshotMb(): Pair<Double, Double> {
        val nativeMb = Debug.getNativeHeapAllocatedSize().toDouble() / (1024.0 * 1024.0)
        val rt = Runtime.getRuntime()
        val javaMb = (rt.totalMemory() - rt.freeMemory()).toDouble() / (1024.0 * 1024.0)
        return Pair(nativeMb, javaMb)
    }

    override fun release() {
        ready = false
    }
}
