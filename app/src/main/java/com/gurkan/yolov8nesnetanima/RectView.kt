package com.gurkan.yolov8nesnetanima

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.round

class RectView(context: Context, attributeSet: AttributeSet) : View(context, attributeSet) {

    private var results: ArrayList<Result>? = null
    private lateinit var classes: Array<String>

    private val textPaint = Paint().also {
        it.textSize = 60f
        it.color = Color.WHITE
    }

    fun setResults(results: ArrayList<Result>) {
        this.results = results
    }




    override fun onDraw(canvas: Canvas?) {
        //sonuçları ekranda dikdörtgen şekilleri olarak çizerek ve her dikdörtgenin içinde nesnenin sınıfının etiketi ve doğruluk oranını yazarak sonuçları görselleştirir.        val scaleX = width / DataProcess.INPUT_SIZE.toFloat()
        val scaleY = scaleX * 9f / 16f
        val realY = width * 9f / 16f
        val diffY = realY - height

        results?.forEach {
            val rectF = it.rectF
            val left = rectF.left * scaleX
            val right = rectF.right * scaleX
            val top = rectF.top * scaleY - (diffY / 2f)
            val bottom = rectF.bottom * scaleY - (diffY / 2f)

            val paint = findPaint(it.classIndex)
            canvas?.drawRect(left, top, right, bottom, paint)
            canvas?.drawText(
                classes[it.classIndex] + ", " + round(it.score * 100) + "%",
                left + 10,
                top + 60,
                textPaint
            )
        }
        super.onDraw(canvas)
    }

    fun setClassLabel(classes: Array<String>) { //tespit edilen nesnelerin etiketlerini ayarlar ve bu etiketleri kullanarak her nesne için özelleştirilmiş bir boya / renk belirler.
        this.classes = classes
    }


    private fun findPaint(classIndex: Int): Paint {
        //sınıf etiketine göre değişen farklı renklerle birlikte bir Paint nesnesi döndürür.
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 10.0f
        paint.strokeCap = Paint.Cap.ROUND
        paint.strokeJoin = Paint.Join.ROUND
        paint.strokeMiter = 100f

        paint.color = when (classIndex) {
            0, 45, 18, 19, 22, 30, 42, 43, 44, 61, 71, 72 -> Color.WHITE
            1, 3, 14, 25, 37, 38, 79 -> Color.BLUE
            2, 9, 10, 11, 32, 47, 49, 51, 52 -> Color.RED
            5, 23, 46, 48 -> Color.YELLOW
            6, 13, 34, 35, 36, 54, 59, 60, 73, 77, 78 -> Color.GRAY
            7, 24, 26, 27, 28, 62, 64, 65, 66, 67, 68, 69, 74, 75 -> Color.BLACK
            12, 29, 33, 39, 41, 58, 50 -> Color.GREEN
            15, 16, 17, 20, 21, 31, 40, 55, 57, 63 -> Color.DKGRAY
            70, 76 -> Color.LTGRAY
            else -> Color.DKGRAY
        }
        return paint
    }
}