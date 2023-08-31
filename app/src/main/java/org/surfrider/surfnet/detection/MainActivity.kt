package org.surfrider.surfnet.detection

import android.app.ActivityManager
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.os.Handler
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.surfrider.surfnet.detection.customview.OverlayView
import org.surfrider.surfnet.detection.customview.OverlayView.DrawCallback
import org.surfrider.surfnet.detection.databinding.ActivityMainBinding
import org.surfrider.surfnet.detection.env.ImageUtils.getTransformationMatrix
import org.surfrider.surfnet.detection.env.Logger
import org.surfrider.surfnet.detection.env.Utils.getBitmapFromAsset
import org.surfrider.surfnet.detection.env.Utils.processBitmap
import org.surfrider.surfnet.detection.tflite.Detector
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import org.surfrider.surfnet.detection.tflite.YoloDetector
import org.surfrider.surfnet.detection.tracking.MultiBoxTracker
import java.io.IOException
import java.util.LinkedList

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val sensorOrientation = 90
    private var detector: Detector? = null
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var tracker: MultiBoxTracker? = null
    private var trackingOverlay: OverlayView? = null
    protected var previewWidth = 0
    protected var previewHeight = 0
    private var sourceBitmap: Bitmap? = null
    private var cropBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.cameraButton.setOnClickListener {
            startActivity(
                Intent(
                    this@MainActivity,
                    DetectorActivity::class.java
                )
            )
        }
        binding.detectButton.setOnClickListener {
            val handler = Handler()
            Thread {
                val results = detector?.recognizeImage(cropBitmap)
                handler.post { handleResult(cropBitmap, results) }
            }.start()
        }
        sourceBitmap = getBitmapFromAsset(this@MainActivity, "kite.jpg")
        cropBitmap = processBitmap(sourceBitmap!!, TF_OD_API_INPUT_SIZE)
        binding.imageView.setImageBitmap(cropBitmap)
        initBox()
        val activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val configurationInfo = activityManager.deviceConfigurationInfo
        System.err.println(configurationInfo.glEsVersion.toDouble())
        System.err.println(configurationInfo.reqGlEsVersion >= 0x30000)
        System.err.println(String.format("%X", configurationInfo.reqGlEsVersion))
    }


    private fun initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE
        previewWidth = TF_OD_API_INPUT_SIZE
        frameToCropTransform = getTransformationMatrix(
            previewWidth, previewHeight,
            TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
            sensorOrientation, MAINTAIN_ASPECT
        )
        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)
        tracker = MultiBoxTracker(this)
        binding.trackingOverlay.addCallback(
            object : DrawCallback {
                override fun drawCallback(canvas: Canvas?) {
                    tracker?.draw(canvas)
                }
            })
        tracker?.setFrameConfiguration(
            TF_OD_API_INPUT_SIZE,
            TF_OD_API_INPUT_SIZE,
            sensorOrientation
        )
        try {
            LOGGER.i("==================== filename: %s", TF_OD_API_MODEL_FILE)
            detector = YoloDetector.create(
                assets,
                TF_OD_API_MODEL_FILE,
                TF_OD_API_LABELS_FILE,
                TF_OD_API_IS_QUANTIZED,
                false,
                TF_OD_API_INPUT_SIZE
            )
        } catch (e: IOException) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing detector!")
            val toast = Toast.makeText(
                applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
    }

    private fun handleResult(bitmap: Bitmap?, results: List<Recognition>?) {
        results?.let {
            val canvas = Canvas(bitmap!!)
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f
            val mappedRecognitions: List<Recognition> = LinkedList()
            for (result in results) {
                val location = result.location
                if (location != null && result.confidence >= MINIMUM_CONFIDENCE_TF_OD_API) {
                    canvas.drawRect(location, paint)
                    //                cropToFrameTransform.mapRect(location);
//
//                result.setLocation(location);
//                mappedRecognitions.add(result);
                }
            }
            //        tracker.trackResults(mappedRecognitions, new Random().nextInt());
//        trackingOverlay.postInvalidate();
            binding.imageView.setImageBitmap(bitmap)
        }
    }

    companion object {
        const val MINIMUM_CONFIDENCE_TF_OD_API = 0.3f
        private val LOGGER = Logger()
        const val TF_OD_API_INPUT_SIZE = 640
        private const val TF_OD_API_IS_QUANTIZED = false
        private const val TF_OD_API_MODEL_FILE = "yolov5sCoco.tflite"
        private const val TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt"

        // Minimum detection confidence to track a detection.
        private const val MAINTAIN_ASPECT = true
    }
}