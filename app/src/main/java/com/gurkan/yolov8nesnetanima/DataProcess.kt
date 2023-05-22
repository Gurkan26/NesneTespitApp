package com.gurkan.yolov8nesnetanima

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.*
import kotlin.math.max
import kotlin.math.min

class DataProcess(val context: Context) {

    lateinit var classes: Array<String>

    companion object {
        const val BATCH_SIZE = 1
        const val INPUT_SIZE = 640
        const val PIXEL_SIZE = 3
        const val FILE_NAME = "yolov8n.onnx"
        const val LABEL_NAME = "yolov8n.txt"
    }

    fun imageToBitmap(imageProxy: ImageProxy): Bitmap { // kamera önizleme görüntüsünü kare olarak yakalıyoruz ve yakaladığımız görüntüyü INPUT_SIZE boyutuna ölçekliyoruz.
        val bitmap = imageProxy.toBitmap()
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
    }

    fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {  // gelen bitmap nesnesinin piksel değerlerini tek boyutlu bir FloatBuffer nesnesine dönüştürüyoruz. Bu işlem sırasında her piksel değeri 0 ile 1 arasında normalleştirilir.
        val imageSTD = 255.0f
        val buffer = FloatBuffer.allocate(BATCH_SIZE * PIXEL_SIZE * INPUT_SIZE * INPUT_SIZE)
        buffer.rewind()

        val area = INPUT_SIZE * INPUT_SIZE
        val bitmapData = IntArray(area)
        bitmap.getPixels(
            bitmapData,
            0,
            bitmap.width,
            0,
            0,
            bitmap.width,
            bitmap.height
        )

        for (i in 0 until INPUT_SIZE - 1) {
            for (j in 0 until INPUT_SIZE - 1) {
                val idx = INPUT_SIZE * i + j
                val pixelValue = bitmapData[idx]
                buffer.put(idx, ((pixelValue shr 16 and 0xff) / imageSTD))
                buffer.put(idx + area, ((pixelValue shr 8 and 0xff) / imageSTD))
                buffer.put(idx + area * 2, ((pixelValue and 0xff) / imageSTD))

            }
        }
        buffer.rewind() // position 0
        return buffer
    }

    fun loadModel() { // eğitilmiş modeli yükler

        val assetManager = context.assets
        val outputFile = File(context.filesDir.toString() + "/" + FILE_NAME)

        assetManager.open(FILE_NAME).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }
    }

    fun loadLabel() {

        BufferedReader(InputStreamReader(context.assets.open(LABEL_NAME))).use { reader ->
            var line: String?
            val classList = ArrayList<String>()
            while (reader.readLine().also { line = it } != null) {
                classList.add(line!!)
            }
            classes = classList.toTypedArray()
        }
    }

    fun outputsToNPMSPredictions(outputs: Array<*>): ArrayList<Result> {
        val confidenceThreshold = 0.45f
        val results = ArrayList<Result>()
        val rows: Int
        val cols: Int

        (outputs[0] as Array<*>).also {
            rows = it.size
            cols = (it[0] as FloatArray).size
        }


        val output = Array(cols) { FloatArray(rows) }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                output[j][i] = ((((outputs[0]) as Array<*>)[i]) as FloatArray)[j]
            }
        }

        for (i in 0 until cols) {
            var detectionClass: Int = -1
            var maxScore = 0f
            val classArray = FloatArray(classes.size)
            System.arraycopy(output[i], 4, classArray, 0, classes.size)
            for (j in classes.indices) {
                if (classArray[j] > maxScore) {
                    detectionClass = j
                    maxScore = classArray[j]
                }
            }

            if (maxScore > confidenceThreshold) {
                val xPos = output[i][0]
                val yPos = output[i][1]
                val width = output[i][2]
                val height = output[i][3]
                val rectF = RectF(
                    max(0f, xPos - width / 2f),
                    max(0f, yPos - height / 2f),
                    min(INPUT_SIZE - 1f, xPos + width / 2f),
                    min(INPUT_SIZE - 1f, yPos + height / 2f)
                )
                val result = Result(detectionClass, maxScore, rectF)
                results.add(result)
            }
        }
        return nms(results)
    }

    private fun nms(results: ArrayList<Result>): ArrayList<Result> {
        val list = ArrayList<Result>()

        for (i in classes.indices) {
            val pq = PriorityQueue<Result>(50) { o1, o2 ->
                o1.score.compareTo(o2.score)
            }
            val classResults = results.filter { it.classIndex == i }
            pq.addAll(classResults)

            while (pq.isNotEmpty()) {
                val detections = pq.toTypedArray()
                val max = detections[0]
                list.add(max)
                pq.clear()

                for (k in 1 until detections.size) {
                    val detection = detections[k]
                    val rectF = detection.rectF
                    val iouThresh = 0.5f
                    if (boxIOU(max.rectF, rectF) < iouThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return list
    }

    private fun boxIOU(a: RectF, b: RectF): Float { //  iki dikdörtgen kutuyu (birincisi a, ikincisi b) kapsayan alanların kesişim oranını hesaplar.
        return boxIntersection(a, b) / boxUnion(a, b)
    }

    private fun boxIntersection(a: RectF, b: RectF): Float { // iki dikdörtgen kutuyu kapsayan alanların kesişim alanını hesaplar.
        val w = overlap( // genişlik hesabı
            (a.left + a.right) / 2f, a.right - a.left,
            (b.left + b.right) / 2f, b.right - b.left
        )
        val h = overlap( // yükseklik hesabı
            (a.top + a.bottom) / 2f, a.bottom - a.top,
            (b.top + b.bottom) / 2f, b.bottom - b.top
        )

        return if (w < 0 || h < 0) 0f else w * h
    }

    private fun boxUnion(a: RectF, b: RectF): Float { //iki dikdörtgen kutuyu kapsayan alanların birleşim alanını hesaplar.
        val i: Float = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    private fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float { //iki aralığın ortak alanını hesaplar ve geri döndürür. Bu metod NMS yönteminde kullanılır.
        // NMS, aynı nesnenin farklı bölgesel teklifleri arasından en güvenilir olanını seçerek, nesne algılama sistemlerinin performansını artırmaya yardımcı olan bir yöntemdir.
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = max(l1, l2)
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = min(r1, r2)
        return right - left
    }
}