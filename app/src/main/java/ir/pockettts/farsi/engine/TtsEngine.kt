package ir.pockettts.farsi.engine

import java.io.File

data class G2pTestCaseResult(
    val id: Int,
    val inputPersian: String,
    val normalizedText: String,
    val expectedPhonemes: String,
    val actualPhonemes: String,
    val passed: Boolean,
    val latencyMs: Long
)

data class MemoryReport(
    val nativeHeapBeforeMb: Double,
    val nativeHeapAfterMb: Double,
    val nativeHeapDeltaMb: Double,
    val javaHeapUsedMb: Double
)

data class SynthesisResult(
    val wavFile: File,
    val normalizedText: String,
    val phonemesWithEzafe: String,
    val chunksRendered: List<String>,
    val totalFrames: Int,
    val sampleRate: Int,
    val audioDurationSec: Double,
    val synthesisTimeSec: Double,
    val rtf: Double,
    val memoryReport: MemoryReport
)

data class ModelFileCheck(
    val fileName: String,
    val exists: Boolean,
    val sizeBytes: Long
)

data class EngineInitResult(
    val ready: Boolean,
    val message: String,
    val modelDir: String,
    val fileChecks: List<ModelFileCheck>
)

/**
 * رابط استاندارد موتور متن به گفتار (مرحله ۱)
 * موتور پیش‌فرض: PocketTtsEngine (موتور Pocket-TTS v2 فارسی مبتنی بر ONNX)
 * موتور اختیاری دوم: KokoroFallbackEngine
 */
interface TtsEngine {
    val engineId: String
    val engineName: String

    fun checkModelDirectory(modelDir: File): List<ModelFileCheck>

    fun initialize(modelDir: File): EngineInitResult

    fun isReady(): Boolean

    fun normalize(text: String): String

    /**
     * تبدیل متن فارسی به رشته فونم.
     * @param keepEzafe اگر true باشد، علامت '1' (کسره اضافه مثل kAble1، qahveye1، poSte1) حفظ می‌شود
     * تا خروجی دقیقاً با ۴ جمله مرجع تست قبولی G2P مقایسه شود.
     */
    fun phonemise(text: String, keepEzafe: Boolean = true): String

    /**
     * اجرای ۴ تست قبولی مرجع G2P و برگرداندن نتیجه مقایسه با مرجع
     */
    fun runG2pAcceptanceSuite(): List<G2pTestCaseResult>

    /**
     * ساخت فایل صوتی WAV ۲۴ کیلوهرتز از متن ورودی و اندازه‌گیری مصرف حافظه Native
     */
    fun synthesizeToWav(
        text: String,
        outputWav: File,
        onStatus: ((String) -> Unit)? = null
    ): SynthesisResult

    fun getMemorySnapshotMb(): Pair<Double, Double>

    fun release()

    companion object {
        val G2P_ACCEPTANCE_REFERENCE = listOf(
            Triple(
                1,
                "کابل را وصل کن",
                "kAbl rA vasl kon"
            ),
            Triple(
                2,
                "قهوه مانده",
                "qahveye1 mAnde"
            ),
            Triple(
                3,
                "کابل شارژر را به گوشی وصل کردم و منتظر ماندم تا صفحه روشن شود.",
                "kAble1 SAr;er rA be guSi vasl kardam va montazer mAndam tA safhe roSan Savad"
            ),
            Triple(
                4,
                "بیرون باران می‌بارید و صدای باران از پشت پنجره شنیده می‌شد.",
                "birun bArAn mibArid va SedAye bArAn ?az poSte1 panjare Senide miSod"
            )
        )
    }
}
