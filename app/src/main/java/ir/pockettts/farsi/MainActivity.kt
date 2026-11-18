package ir.pockettts.farsi

import android.Manifest
import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import ir.pockettts.farsi.engine.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {

    private val pocketEngine: TtsEngine = PocketTtsEngine()
    private val kokoroSecondaryEngine: TtsEngine = KokoroFallbackEngine()
    private var mediaPlayer: MediaPlayer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestStoragePermissionsIfNeeded()

        setContent {
            CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
                MaterialTheme(
                    colorScheme = darkColorScheme(
                        primary = Color(0xFFF59E0B),
                        secondary = Color(0xFF10B981),
                        background = Color(0xFF0A0A0A),
                        surface = Color(0xFF171717)
                    )
                ) {
                    Surface(
                        modifier = Modifier.fillMaxSize(),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        Stage1TestScreen(
                            defaultEngine = pocketEngine,
                            secondaryEngine = kokoroSecondaryEngine,
                            cacheDir = cacheDir,
                            onPlayWav = { wavFile -> playWavFile(wavFile) }
                        )
                    }
                }
            }
        }
    }

    private fun playWavFile(wavFile: File) {
        try {
            mediaPlayer?.release()
            mediaPlayer = MediaPlayer().apply {
                setDataSource(wavFile.absolutePath)
                prepare()
                start()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun requestStoragePermissionsIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                try {
                    val intent = Intent(
                        Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:$packageName")
                    )
                    startActivity(intent)
                } catch (_: Exception) {
                    val intent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                    startActivity(intent)
                }
            }
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                ),
                1001
            )
        }
    }

    override fun onDestroy() {
        mediaPlayer?.release()
        pocketEngine.release()
        kokoroSecondaryEngine.release()
        super.onDestroy()
    }
}

@Composable
fun Stage1TestScreen(
    defaultEngine: TtsEngine,
    secondaryEngine: TtsEngine,
    cacheDir: File,
    onPlayWav: (File) -> Unit
) {
    val scope = rememberCoroutineScope()
    var useDefaultPocketEngine by remember { mutableStateOf(true) }
    val activeEngine = if (useDefaultPocketEngine) defaultEngine else secondaryEngine

    var modelDirPath by remember {
        mutableStateOf(
            File(Environment.getExternalStorageDirectory(), "PocketTTS/model").absolutePath
        )
    }
    var initResult by remember { mutableStateOf<EngineInitResult?>(null) }
    var g2pResults by remember { mutableStateOf<List<G2pTestCaseResult>>(emptyList()) }
    var inputSentence by remember {
        mutableStateOf("کابل شارژر را به گوشی وصل کردم و منتظر ماندم تا صفحه روشن شود.")
    }
    var livePhonemes by remember { mutableStateOf("") }
    var statusMessage by remember { mutableStateOf("آماده بارگذاری مدل از حافظه گوشی") }
    var isBusy by remember { mutableStateOf(false) }
    var lastSynthesis by remember { mutableStateOf<SynthesisResult?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "کتاب صوتی فارسی — مرحله ۱ (تست موتور Pocket-TTS v2)",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFFFBBF24)
        )

        // ۱. انتخاب موتور (پیش‌فرض: موتور Pocket-TTS v2 فارسی کاربر)
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF171717))
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("انتخاب موتور (Interface: TtsEngine):", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { useDefaultPocketEngine = true },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (useDefaultPocketEngine) Color(0xFFF59E0B) else Color(0xFF262626),
                            contentColor = if (useDefaultPocketEngine) Color.Black else Color.White
                        )
                    ) {
                        Text("Pocket-TTS v2 فارسی (پیش‌فرض)")
                    }
                    OutlinedButton(onClick = { useDefaultPocketEngine = false }) {
                        Text("Kokoro (گزینه دوم)")
                    }
                }

                OutlinedTextField(
                    value = modelDirPath,
                    onValueChange = { modelDirPath = it },
                    label = { Text("مسیر پوشه مدل‌ها روی حافظه گوشی (بدون قرارگیری در assets)") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Button(
                    enabled = !isBusy,
                    onClick = {
                        scope.launch {
                            isBusy = true
                            statusMessage = "در حال بررسی و بارگذاری مدل‌ها..."
                            val res = withContext(Dispatchers.IO) {
                                activeEngine.initialize(File(modelDirPath))
                            }
                            initResult = res
                            statusMessage = res.message
                            isBusy = false
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("۱. بارگذاری مدل‌ها از حافظه گوشی (ایجاد یک‌باره Sessionها)")
                }

                Text(text = statusMessage, fontSize = 12.sp, color = Color(0xFFD4D4D8))

                initResult?.let { res ->
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        res.fileChecks.forEach { chk ->
                            val badge = if (chk.exists) "✅ موجود" else "❌ غایب"
                            val mb = "%.2f MB".format(chk.sizeBytes / (1024.0 * 1024.0))
                            Text(
                                text = "$badge  ${chk.fileName} ($mb)",
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                color = if (chk.exists) Color(0xFF34D399) else Color(0xFFF87171)
                            )
                        }
                    }
                }
            }
        }

        // ۲. تست قبولی G2P روی ۴ جمله مرجع
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF171717))
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    text = "۲. تست قبولی G2P (مقایسه دقیق فونم ۴ جمله مرجع)",
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp,
                    color = Color(0xFF34D399)
                )

                Button(
                    enabled = !isBusy,
                    onClick = {
                        scope.launch {
                            isBusy = true
                            if (!activeEngine.isReady()) {
                                initResult = withContext(Dispatchers.IO) {
                                    activeEngine.initialize(File(modelDirPath))
                                }
                            }
                            g2pResults = withContext(Dispatchers.IO) {
                                activeEngine.runG2pAcceptanceSuite()
                            }
                            isBusy = false
                        }
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF10B981),
                        contentColor = Color.Black
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("اجرای تست قبولی G2P و چاپ فونم‌های هر ۴ جمله")
                }

                g2pResults.forEach { item ->
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF0A0A0A), RoundedCornerShape(12.dp))
                            .padding(10.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "جمله #${item.id}: ${item.inputPersian}",
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 13.sp,
                                color = Color.White
                            )
                            Text(
                                text = if (item.passed) "✅ تطابق دقیق (PASS)" else "⚠️ عدم تطابق (DIFF)",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (item.passed) Color(0xFF34D399) else Color(0xFFFBBF24)
                            )
                        }
                        CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Ltr) {
                            Text(
                                text = "REF: ${item.expectedPhonemes}",
                                fontFamily = FontFamily.Monospace,
                                fontSize = 12.sp,
                                color = Color(0xFF9CA3AF)
                            )
                            Text(
                                text = "OUT: ${item.actualPhonemes}",
                                fontFamily = FontFamily.Monospace,
                                fontSize = 12.sp,
                                color = if (item.passed) Color(0xFF34D399) else Color(0xFFF87171)
                            )
                        }
                    }
                }
            }
        }

        // ۳. تایپ جمله، ساخت صدا با موتور Pocket-TTS v2 و پخش + گزارش حافظه
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF171717))
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    text = "۳. ساخت صدا و پخش (FlowLM + Mimi Decoder 24kHz)",
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp,
                    color = Color(0xFFFBBF24)
                )

                OutlinedTextField(
                    value = inputSentence,
                    onValueChange = { inputSentence = it },
                    label = { Text("جمله فارسی را وارد کنید") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        enabled = !isBusy,
                        onClick = {
                            scope.launch {
                                isBusy = true
                                if (!activeEngine.isReady()) {
                                    initResult = withContext(Dispatchers.IO) {
                                        activeEngine.initialize(File(modelDirPath))
                                    }
                                }
                                livePhonemes = withContext(Dispatchers.IO) {
                                    activeEngine.phonemise(inputSentence, keepEzafe = true)
                                }
                                isBusy = false
                            }
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("فقط استخراج فونم")
                    }

                    Button(
                        enabled = !isBusy,
                        onClick = {
                            scope.launch {
                                isBusy = true
                                try {
                                    if (!activeEngine.isReady()) {
                                        initResult = withContext(Dispatchers.IO) {
                                            activeEngine.initialize(File(modelDirPath))
                                        }
                                    }
                                    val outWav = File(cacheDir, "stage1_output.wav")
                                    val res = withContext(Dispatchers.IO) {
                                        activeEngine.synthesizeToWav(inputSentence, outWav) { progress ->
                                            statusMessage = progress
                                        }
                                    }
                                    livePhonemes = res.phonemesWithEzafe
                                    lastSynthesis = res
                                    statusMessage = "ساخت صدا کامل شد. در حال پخش..."
                                    onPlayWav(res.wavFile)
                                } catch (e: Exception) {
                                    statusMessage = "خطا: ${e.message}"
                                } finally {
                                    isBusy = false
                                }
                            }
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("ساخت صدا و پخش ▶")
                    }
                }

                if (livePhonemes.isNotEmpty()) {
                    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Ltr) {
                        Text(
                            text = "Phonemes: $livePhonemes",
                            fontFamily = FontFamily.Monospace,
                            fontSize = 13.sp,
                            color = Color(0xFF38BDF8),
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(Color(0xFF0A0A0A), RoundedCornerShape(10.dp))
                                .padding(10.dp)
                        )
                    }
                }

                lastSynthesis?.let { syn ->
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF0A0A0A), RoundedCornerShape(12.dp))
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text("گزارش عملکرد و حافظه Native (مرحله ۱):", fontWeight = FontWeight.Bold, color = Color(0xFF34D399))
                        Text("• مدت زمان صدا: %.2f ثانیه (%d فریم)".format(syn.audioDurationSec, syn.totalFrames), fontSize = 12.sp)
                        Text("• زمان تولید: %.2f ثانیه | ضریب سرعت (RTF): %.2fx".format(syn.synthesisTimeSec, syn.rtf), fontSize = 12.sp)
                        Text(
                            "• حافظه Native قبل: %.1f MB | بعد: %.1f MB | تغییرات (Delta): %+.2f MB".format(
                                syn.memoryReport.nativeHeapBeforeMb,
                                syn.memoryReport.nativeHeapAfterMb,
                                syn.memoryReport.nativeHeapDeltaMb
                            ),
                            fontSize = 12.sp,
                            color = Color(0xFFFBBF24)
                        )
                        Text("• حافظه Java Heap مصرفی: %.1f MB".format(syn.memoryReport.javaHeapUsedMb), fontSize = 12.sp)
                        Button(
                            onClick = { onPlayWav(syn.wavFile) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("پخش مجدد فایل WAV ساخته شده")
                        }
                    }
                }
            }
        }
    }
}
