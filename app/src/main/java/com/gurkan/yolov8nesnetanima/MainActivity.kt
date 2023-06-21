package com.gurkan.yolov8nesnetanima

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import com.google.android.gms.vision.Frame
import com.google.android.gms.vision.text.TextRecognizer
import java.util.*
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var rectView: RectView
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var session: OrtSession


    private val dataProcess = DataProcess(context = this)

    companion object {
        const val PERMISSION = 1
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        rectView = findViewById(R.id.rectView)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)


        setPermissions()

        load()

        setCamera()
    }
    private fun setPermissions() {// kullanıcının kamerasını kullanmak için izin istediğimiz yer.
        val permissions = ArrayList<String>()
        permissions.add(android.Manifest.permission.CAMERA)

        permissions.forEach {
            if (ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions.toTypedArray(), PERMISSION)
            }
        }
    }

    private fun setCamera() { //Kamera önizlemesi için gerekli yapılandırmaları yapıyoruz. Önizleme boyutları belirleyip ve uygun bir yüzey sağlıyoruz.
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        previewView.post { // dOĞRU Çerçevelenemediği için preview boyutunu ayarladığımız kısım

            val parentWidth = previewView.parent as View
            val width = parentWidth.width
            previewView.layoutParams.height = (width * 9f / 16f).toInt()
        }
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_DEFAULT).build()

        preview.setSurfaceProvider(previewView.surfaceProvider)

        val analysis = ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()

        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            imageProcess(it)
            it.close()
        }

        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }


    private fun imageProcess(imageProxy: ImageProxy) {
        val bitmap = dataProcess.imageToBitmap(imageProxy)
        // OCR işlemi için metnin tanınacağı bitmap'i oluşturun.
        val textBitmap = Bitmap.createBitmap(bitmap)
        recognizeText(textBitmap)

        val floatBuffer = dataProcess.bitmapToFloatBuffer(bitmap)
        val inputName = session.inputNames.iterator().next()
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
        val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))
        val outputs = resultTensor.get(0).value as Array<*>
        val results = dataProcess.outputsToNPMSPredictions(outputs)

        runOnUiThread {
            rectView.setResults(results)
            rectView.invalidate()
        }

        imageProxy.close()
    }


    private fun load() { // labellerin ve modelin yüklendiği kısım burası
        dataProcess.loadModel()
        dataProcess.loadLabel()

        ortEnvironment = OrtEnvironment.getEnvironment()
        session = ortEnvironment.createSession(
            this.filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME,
            OrtSession.SessionOptions()
        )

        rectView.setClassLabel(dataProcess.classes)
    }

    override fun onRequestPermissionsResult( // Kullanıcıdan izin alınıp alınmadığını kontrol ettiğimış yer.
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == PERMISSION) {
            grantResults.forEach {
                if (it != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Kamera İzni Vermeniz Gerekiyor.", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun recognizeText(bitmap: Bitmap) {
        val textRecognizer = TextRecognizer.Builder(this).build()
        val textViewYazi= findViewById<TextView>(R.id.textviewYazi)
        if (!textRecognizer.isOperational) {
            // Tesseract OCR henüz hazır değilse, hata mesajı yazdırılır.
            runOnUiThread {
                Toast.makeText(applicationContext, "OCR not operational.", Toast.LENGTH_SHORT).show()
            }
        } else {
            // Tesseract OCR kullanarak metni okur.
            val frame = Frame.Builder().setBitmap(bitmap).build()
            val textBlocks = textRecognizer.detect(frame)

            if (textBlocks.size() == 0) {
                runOnUiThread {
                    textViewYazi.text = ""
                }
            } else {
                val stringBuilder = StringBuilder()

                for (i in 0 until textBlocks.size()) {
                    stringBuilder.append(textBlocks[i].value)
                    stringBuilder.append("\n")
                }

                val recognizedText = stringBuilder.toString()

                // Metin işleme için özelleştirilmiş modelinizi burada kullanarak,
                // recognizedText üzerinde işlem yapabilirsiniz.

                runOnUiThread {
                    textViewYazi.text = recognizedText
                }

                val floatBuffer = dataProcess.bitmapToFloatBuffer(bitmap)
                val inputName = session.inputNames.iterator().next()
                val shape = longArrayOf(
                    DataProcess.BATCH_SIZE.toLong(),
                    DataProcess.PIXEL_SIZE.toLong(),
                    DataProcess.INPUT_SIZE.toLong(),
                    DataProcess.INPUT_SIZE.toLong()
                )
                val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
                val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))
                val outputs = resultTensor.get(0).value as Array<*>
                val results = dataProcess.outputsToNPMSPredictions(outputs)

                runOnUiThread {
                    rectView.setResults(results)
                    rectView.invalidate()
                }
            }
        }
    }






}