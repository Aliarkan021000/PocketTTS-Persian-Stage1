package ir.pockettts.farsi.engine

import java.io.File

/**
 * توکن‌ساز BPE دقیقاً منطبق با java/GOLD_20261031/TokTest.java
 * تست شده روی ۲۰۰/۲۰۰ جمله طلایی (tok_gold.tsv)
 */
class VocabTokenizer(vocabPhTsv: File) {
    private val pieceToId = HashMap<String, Int>(4096)
    private val pieceToScore = HashMap<String, Float>(4096)

    init {
        require(vocabPhTsv.exists()) { "فایل vocab_ph.tsv در مسیر ${vocabPhTsv.absolutePath} یافت نشد." }
        vocabPhTsv.forEachLine(Charsets.UTF_8) { line ->
            if (line.isNotEmpty()) {
                val parts = line.split('\t')
                if (parts.size >= 3) {
                    val id = parts[0].toInt()
                    val piece = parts[1]
                    val score = parts[2].toFloat()
                    pieceToId[piece] = id
                    pieceToScore[piece] = score
                }
            }
        }
    }

    fun encode(text: String): IntArray {
        val clean = text.trim()
        if (clean.isEmpty()) return IntArray(0)
        val normalized = "\u2581" + clean.replace(" ", "\u2581")

        val syms = ArrayList<String>(normalized.length)
        var i = 0
        while (i < normalized.length) {
            val cp = normalized.codePointAt(i)
            syms.add(String(Character.toChars(cp)))
            i += Character.charCount(cp)
        }

        while (syms.size > 1) {
            var bestIdx = -1
            var bestScore = Float.NEGATIVE_INFINITY
            for (j in 0 until syms.size - 1) {
                val merged = syms[j] + syms[j + 1]
                val sc = pieceToScore[merged]
                if (sc != null && sc > bestScore) {
                    bestScore = sc
                    bestIdx = j
                }
            }
            if (bestIdx < 0) break
            syms[bestIdx] = syms[bestIdx] + syms[bestIdx + 1]
            syms.removeAt(bestIdx + 1)
        }

        val ids = IntArray(syms.size)
        for (k in syms.indices) {
            ids[k] = pieceToId[syms[k]] ?: 0
        }
        return ids
    }

    fun nTok(text: String): Int = encode(text.replace("1", "")).size
}

/**
 * چانکر هوشمند فونم‌ها منطبق با java/GOLD_20261031/Chunker.java و engine/tts_onnx.py
 * به همراه ادغام جملات خیلی کوتاه (۳ تا ۴ کلمه) با جمله کناری
 */
object PhonemeChunker {
    private const val CT = 18
    private const val HR = 2
    private const val TMC = 19
    private const val MINW = 3
    private const val MINT = 6
    private const val RUNT = 2
    private const val LBLV = 5
    private const val LBP = 3

    private val LIGHT = setOf(
        "kard", "kardam", "kardi", "kardim", "kardid", "kardand",
        "mikard", "mikardam", "mikardand", "kon", "konam", "koni", "konad", "konim", "konid", "konand",
        "mikonam", "mikoni", "mikonad", "mikonim", "mikonand", "bekon", "nakon", "nakard",
        "Sod", "Sodam", "Sodi", "Sodim", "Sodand", "miSavad", "miSod", "Savad", "Savand", "beSavad", "naSod",
        "bud", "budam", "budi", "budim", "budand", "bASad", "bASam", "bASi", "bASim", "bASand",
        "hast", "hastam", "hasti", "hastim", "hastand", "nist", "nistam", "nisti",
        "dASht", "dASt", "dAStam", "dASti", "dAStim", "dAStand",
        "dArad", "dAram", "dAri", "dArim", "dArand", "midArad", "nadArad", "nadASt",
        "zadan", "zad", "zadam", "zadand", "mizanad", "mizanand", "bezan",
        "dAd", "dAdam", "dAdand", "midahad", "midahand", "bedeh",
        "gereft", "gereftam", "gereftand", "migirad", "migirand", "begir",
        "raft", "raftam", "raftand", "miravad", "miravand", "bero",
        "?Amad", "?Amadam", "?Amadand", "mi?Ayad", "bi?A",
        "xord", "xordam", "xordand", "mixorad",
        "ديد", "did", "didam", "didand", "mibinad",
        "goft", "goftam", "goftand", "miguyad",
        "xAst", "xAstam", "mixAhad", "mixAham",
        "tavAnest", "mitavAnad", "mitavAnam", "bAyad", "SAyad"
    )

    private val PREPS = setOf(
        "?az", "be", "bA", "dar", "bar", "tA", "bi", "joz",
        "barAye", "barAye1", "bedune", "bedune1", "mesle", "mesle1",
        "mAnande", "mAnande1", "ruye", "ruye1", "zire", "zire1",
        "piSe", "piSe1", "nazde", "nazde1", "pahlu", "jebheye", "jebheye1",
        "kenAre", "kenAre1", "dAxzele", "dAxele1", "birune", "birune1",
        "bAlAye", "bAlAye1", "pA?ine", "pA?ine1", "poSte", "poSte1",
        "jeloye", "jeloye1", "vasate", "vasate1", "miyAne", "miyAne1",
        "dor", "dore", "dore1", "?aTrAfe", "?aTrAfe1", "darbAreye", "darbAreye1",
        "bAbate", "bAbate1", "nesbat", "طبق", "tebqe", "tebqe1", "علیه", "?alayhe"
    )

    private val FUNC = PREPS + setOf("rA", "ham", "niz", "ke", "va", "yA", "?ammA", "vali", "?agar", "pas", "con")
    private val CONJ = setOf("va", "yA", "?ammA", "vali", "ke", "con", "zirA", "pas", "sepas", "hattA", "balkeh", "balke")

    private fun clean(w: String): String = w.trim('.', '،', '؛', '؟', '!', ':', ',', ';', '?')
    private fun bare(w: String): String = clean(w).removeSuffix("1")
    private fun hasEz(w: String): Boolean = clean(w).endsWith("1")
    private fun hasPunct(w: String): Boolean {
        if (w.isEmpty()) return false
        return w.last() in ".،؛؟!:,;?"
    }
    private fun isLight(w: String): Boolean = bare(w) in LIGHT
    private fun isPrep(w: String): Boolean {
        val c = clean(w)
        return c in PREPS || c.removeSuffix("1") in PREPS
    }
    private fun isFunc(w: String): Boolean {
        val c = clean(w)
        return c in FUNC || c.removeSuffix("1") in FUNC
    }

    /**
     * درس مهم کاربر:
     * «ورودی خیلی کوتاه (۳–۴ کلمه) گاهی رفتار عجیب دارد. جمله‌های کوتاه رو با جمله‌ی کناری یکی کن.»
     */
    fun mergeShortPersianSentences(rawParagraph: String, minWords: Int = 5): List<String> {
        val rawSentences = rawParagraph
            .trim()
            .split(Regex("(?<=[.؟!؛])\\s+"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        if (rawSentences.size <= 1) return rawSentences

        val merged = mutableListOf<String>()
        var buffer = ""

        for (sent in rawSentences) {
            buffer = if (buffer.isEmpty()) sent else "$buffer $sent"
            val wordCount = buffer.split(Regex("\\s+")).count { it.isNotBlank() }
            if (wordCount >= minWords) {
                merged.add(buffer)
                buffer = ""
            }
        }
        if (buffer.isNotEmpty()) {
            if (merged.isNotEmpty()) {
                merged[merged.lastIndex] = merged.last() + " " + buffer
            } else {
                merged.add(buffer)
            }
        }
        return merged
    }

    fun planPhrases(ph: String): List<List<String>> {
        val words = ph.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.isEmpty()) return emptyList()

        val groups = ArrayList<MutableList<String>>()
        var i = 0
        while (i < words.size) {
            val grp = mutableListOf(words[i])
            while (
                i + 1 < words.size &&
                !hasPunct(words[i]) &&
                (hasEz(words[i]) || isPrep(words[i]) || bare(words[i + 1]) == "rA")
            ) {
                i++
                grp.add(words[i])
            }
            if (i + 1 < words.size && !hasPunct(words[i]) && isLight(words[i + 1])) {
                i++
                grp.add(words[i])
                while (i + 1 < words.size && !hasPunct(words[i]) && isLight(words[i + 1])) {
                    i++
                    grp.add(words[i])
                }
            }
            groups.add(grp)
            i++
        }

        val phrases = ArrayList<MutableList<String>>()
        var cur = mutableListOf<String>()
        for (g in groups) {
            val gFirst = bare(g[0])
            if (cur.isNotEmpty() && gFirst in CONJ && cur.size >= MINW) {
                phrases.add(cur)
                cur = ArrayList(g)
            } else {
                cur.addAll(g)
            }
            if (hasPunct(g.last()) && cur.size >= MINW) {
                phrases.add(cur)
                cur = mutableListOf()
            }
        }
        if (cur.isNotEmpty()) {
            if (phrases.isNotEmpty() && cur.size < MINW) {
                phrases.last().addAll(cur)
            } else {
                phrases.add(cur)
            }
        }
        return phrases
    }

    private fun splitLongPhrase(phrase: List<String>, tok: VocabTokenizer, budget: Int): List<List<String>> {
        val out = ArrayList<List<String>>()
        var cur = mutableListOf<String>()
        var i = 0
        while (i < phrase.size) {
            val unit = mutableListOf(phrase[i])
            while (
                i + 1 < phrase.size &&
                (hasEz(phrase[i]) || isPrep(phrase[i]) || bare(phrase[i + 1]) == "rA" || isLight(phrase[i + 1]))
            ) {
                i++
                unit.add(phrase[i])
            }
            val cand = ArrayList(cur).apply { addAll(unit) }
            if (cur.isNotEmpty() && tok.nTok(cand.joinToString(" ")) > budget) {
                out.add(cur)
                cur = ArrayList(unit)
            } else {
                cur = cand
            }
            i++
        }
        if (cur.isNotEmpty()) out.add(cur)
        return out
    }

    fun chunkPhonemes(ph: String, tok: VocabTokenizer): List<String> {
        val phrases = planPhrases(ph)
        if (phrases.isEmpty()) return emptyList()

        val budget = CT - HR
        val expanded = ArrayList<List<String>>()
        for (p in phrases) {
            if (tok.nTok(p.joinToString(" ")) <= budget) {
                expanded.add(p)
            } else {
                expanded.addAll(splitLongPhrase(p, tok, budget))
            }
        }

        val chunks = ArrayList<MutableList<String>>()
        var cur = mutableListOf<String>()
        for (p in expanded) {
            if (cur.isEmpty()) {
                cur = ArrayList(p)
                continue
            }
            val cand = ArrayList(cur).apply { addAll(p) }
            val candT = tok.nTok(cand.joinToString(" "))
            val curT = tok.nTok(cur.joinToString(" "))
            val curEndsSent = hasPunct(cur.last()) && cur.last().last() in ".؟!?;"
            if (candT <= budget && !(curEndsSent && curT >= MINT)) {
                cur = cand
            } else {
                chunks.add(cur)
                cur = ArrayList(p)
            }
        }
        if (cur.isNotEmpty()) chunks.add(cur)

        // ادغام چانک‌های خیلی کوچک (Runt Merge)
        var idx = 0
        while (idx < chunks.size) {
            val c = chunks[idx]
            val ct = tok.nTok(c.joinToString(" "))
            if ((c.size <= RUNT || ct < MINT) && chunks.size > 1) {
                if (idx > 0) {
                    val prev = ArrayList(chunks[idx - 1]).apply { addAll(c) }
                    if (tok.nTok(prev.joinToString(" ")) <= TMC) {
                        chunks[idx - 1] = prev
                        chunks.removeAt(idx)
                        continue
                    }
                }
                if (idx + 1 < chunks.size) {
                    val nxt = ArrayList(c).apply { addAll(chunks[idx + 1]) }
                    if (tok.nTok(nxt.joinToString(" ")) <= TMC) {
                        chunks[idx + 1] = nxt
                        chunks.removeAt(idx)
                        continue
                    }
                }
            }
            idx++
        }

        return chunks
            .map { chunk -> chunk.joinToString(" ").replace("1", "").trim() }
            .filter { it.isNotEmpty() }
    }
}
