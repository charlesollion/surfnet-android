/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.surfrider.surfnet.detection


import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.ImageReader.OnImageAvailableListener
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.SystemClock
import android.util.Size
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.core.app.ActivityCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import org.surfrider.surfnet.detection.customview.OverlayView
import org.surfrider.surfnet.detection.customview.OverlayView.DrawCallback
import org.surfrider.surfnet.detection.env.ImageUtils.getTransformationMatrix
import org.surfrider.surfnet.detection.env.ImageUtils.saveBitmap
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import org.surfrider.surfnet.detection.tflite.YoloDetector
import org.surfrider.surfnet.detection.tracking.TrackerManager
import timber.log.Timber
import java.io.IOException
import java.util.LinkedList


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
class DetectorActivity : CameraActivity(), OnImageAvailableListener, LocationListener {
    private var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int = 0
    private var detector: YoloDetector? = null
    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var cropCopyBitmap: Bitmap? = null
    private var computingDetection = false
    private var timestamp: Long = 0
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var trackerManager: TrackerManager? = null
    private lateinit var locationManager: LocationManager
    private val modelString = "yolov8n_float16.tflite"
    private val labelFilename = "file:///android_asset/coco.txt"
    private val inputSize = 640
    private val isV8 = true
    private val isQuantized = false
    private val numThreads = 1
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private val locationHandler = Handler()
    private var lastTrackerManager: TrackerManager? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
               hideSystemUI()
                fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
           updateLocation()

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        updateLocation()
    }
    private fun hideSystemUI() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            window.setFlags(
                    WindowManager.LayoutParams.FLAG_FULLSCREEN,
                    WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            hideNavigationBar()
        } else {
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        }
    }
    @RequiresApi(Build.VERSION_CODES.R)
    private fun hideNavigationBar() {
        val windowInsetsController =
                WindowCompat.getInsetsController(window, window.decorView)
        windowInsetsController.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        window.decorView.setOnApplyWindowInsetsListener { view, windowInsets ->
            if (windowInsets.isVisible(WindowInsetsCompat.Type.navigationBars())
                    || windowInsets.isVisible(WindowInsetsCompat.Type.statusBars())) {
            }
            view.onApplyWindowInsets(windowInsets)
        }
    }
    private fun getLocation() {
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if ((ActivityCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED) && (ActivityCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED)
        ) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), REQUEST_LOCATION_PERMISSION
            )
            return
        }
        val location = fusedLocationClient.lastLocation
        location.addOnSuccessListener {
            if (it != null) {
                showGPSCoordinates(arrayOf(it.longitude.toString(), it.latitude.toString()))
            }
        }
    }

    private fun updateLocation() {
        locationHandler.postDelayed({
            getLocation()
            locationHandler.postDelayed({
                updateLocation()
            }, 1000)
        }, 0)
    }

    override fun onLocationChanged(location: Location) {
        getLocation()
    }

    override fun onDestroy() {
        super.onDestroy()
        locationHandler.removeCallbacksAndMessages(null)
    }

    public override fun endDetector() {
        trackerManager?.let {
            tracker -> lastTrackerManager = tracker
        }
        detectorPaused = true
        //removes all drawings of the trackingOverlay from the screen
        trackingOverlay?.invalidate()

        //reset trackers
        croppedBitmap = null
        cropToFrameTransform = null
        trackerManager = null
        trackingOverlay = null
    }

    public override fun startDetector() {
        val context = this
        try {
            //create detector only one time
            if (detector == null) {
                detector = YoloDetector.create(
                    assets, modelString, labelFilename, isQuantized, isV8, inputSize
                )
            }
            detector?.useGpu()
            detector?.setNumThreads(numThreads)
            detectorPaused = false
        } catch (e: IOException) {
            e.printStackTrace()
            Timber.e(e, "Exception initializing Detector!")
            val toast = Toast.makeText(
                applicationContext, "Detector could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
        trackerManager = if (lastTrackerManager != null)
            lastTrackerManager
        else
            TrackerManager()
        val cropSize = detector?.inputSize
        cropSize?.let {
            Timber.i(Bitmap.createBitmap(it, it, Bitmap.Config.ARGB_8888).toString())
            croppedBitmap = Bitmap.createBitmap(it, it, Bitmap.Config.ARGB_8888)
            frameToCropTransform = getTransformationMatrix(
                previewWidth, previewHeight, it, it, sensorOrientation, MAINTAIN_ASPECT
            )
        }

        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)
        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay?.addCallback(object : DrawCallback {
            override fun drawCallback(canvas: Canvas?) {
                if (canvas != null) {
                    trackerManager?.draw(canvas, context, previewWidth, previewHeight)
                }
                if (isDebug) {
                    if (canvas != null) {
                        trackerManager?.drawDebug(canvas)
                    }
                }
                trackerManager?.let {
                   tracker -> updateCounter(tracker.detectedWaste.size)
                }

                //drawDebugScreen(canvas)
            }
        })
    }

    fun drawDebugScreen(canvas: Canvas?) {
        // Debug function to show frame size and crop size
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4.0f
        val rectCrop = RectF(0.0F, 0.0F, 640.0F, 640.0F)
        // Slightly smaller than Camera frame width to see all borders
        val rectCam = RectF(90.0F, 90.0F, previewWidth.toFloat()-90.0F, previewHeight.toFloat()-90.0F)

        val frameToCanvasTransform = Matrix()
        val scale = Math.min(canvas!!.width / previewWidth.toFloat(), canvas!!.height / previewHeight.toFloat())
        frameToCanvasTransform.postScale(scale, scale)

        // Draw Camera frame
        frameToCanvasTransform.mapRect(rectCam)
        canvas?.drawRect(rectCam, paint)

        // Draw Crop
        paint.color = Color.GREEN
        cropToFrameTransform?.mapRect(rectCrop)
        frameToCanvasTransform.mapRect(rectCrop)
        canvas?.drawRect(rectCrop, paint)
    }

    public override fun onPreviewSizeChosen(size: Size?, rotation: Int?) {
        trackerManager = TrackerManager()

        size?.let {
            previewWidth = it.width
            previewHeight = it.height
        }
        if (rotation != null) {
            sensorOrientation = rotation - screenOrientation
        }
        Timber.i("Camera orientation relative to screen canvas: %d", sensorOrientation)
        Timber.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
    }


    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        trackingOverlay?.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true

        // Timber.i("Preparing image $currTimestamp for detection in bg thread.")
        getRgbBytes()?.let {
            rgbFrameBitmap?.setPixels(
                it, 0, previewWidth, 0, 0, previewWidth, previewHeight
            )
        }
        readyForNextImage()
        if (croppedBitmap != null && rgbFrameBitmap != null && frameToCropTransform != null) {
            val canvas = Canvas(croppedBitmap!!)
            canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)
            // For examining the actual TF input.
            if (SAVE_PREVIEW_BITMAP) {
                saveBitmap(croppedBitmap!!)
            }
        }
        runInBackground {
            // Timber.i("Running detection on image $currTimestamp")
            val startTime = SystemClock.uptimeMillis()
            val results: List<Recognition>? = croppedBitmap?.let {

                detector?.let {
                    it.recognizeImage(croppedBitmap)
                }
            }

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            cropCopyBitmap = croppedBitmap?.let { Bitmap.createBitmap(it) }
            val canvas = cropCopyBitmap?.let { Canvas(it) }
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f
            val minimumConfidence: Float = when (MODE) {
                DetectorMode.TF_OD_API -> MINIMUM_CONFIDENCE_TF_OD_API
            }
            val mappedRecognitions: MutableList<Recognition> = LinkedList()
            if (results != null) {
                for (result in results) {
                    val location = result.location
                    if (location != null && result.confidence >= minimumConfidence) {
                        canvas?.drawRect(location, paint)
                        cropToFrameTransform?.mapRect(location)
                        result.location = location
                        mappedRecognitions.add(result)
                    }
                }
            }
            trackerManager?.processDetections(mappedRecognitions)
            trackingOverlay?.postInvalidate()
            computingDetection = false
            runOnUiThread {
                showInference(lastProcessingTimeMs.toString() + "ms")
            }
        }
    }

    override val layoutId: Int
        get() = R.layout.tfe_od_camera_connection_fragment_tracking
    override val desiredPreviewFrameSize: Size?
        get() = Size(1280, 720)

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum class DetectorMode {
        TF_OD_API
    }

    companion object {
        private val MODE = DetectorMode.TF_OD_API
        const val MINIMUM_CONFIDENCE_TF_OD_API = 0.3f
        private const val MAINTAIN_ASPECT = true
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
        private const val REQUEST_LOCATION_PERMISSION = 2
    }
}