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
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.hardware.camera2.CameraManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Size
import android.view.Surface
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.TableRow
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import org.surfrider.surfnet.detection.customview.BottomSheet
import org.surfrider.surfnet.detection.customview.OverlayView
import org.surfrider.surfnet.detection.databinding.TfeOdActivityCameraBinding
import org.surfrider.surfnet.detection.env.ImageProcessor
import org.surfrider.surfnet.detection.env.ImageUtils
import org.surfrider.surfnet.detection.env.Utils.chooseCamera
import org.surfrider.surfnet.detection.tflite.Detector
import org.surfrider.surfnet.detection.tflite.YoloDetector
import org.surfrider.surfnet.detection.tracking.TrackerManager
import timber.log.Timber
import java.io.IOException


class TrackingActivity : AppCompatActivity(), OnImageAvailableListener, LocationListener {

    private lateinit var locationManager: LocationManager
    private lateinit var binding: TfeOdActivityCameraBinding
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var bottomSheet: BottomSheet
    private lateinit var chronometer: TableRow
    private lateinit var imageProcessor: ImageProcessor

    private var previewWidth = 0

    private var previewHeight = 0
    private var handler: Handler? = null
    private var handlerThread: HandlerThread? = null
    private var detectorPaused = true
    private var wasteCount = 0
    private var location: Location? = null

    private var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int = 0
    private var detector: YoloDetector? = null
    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var computingDetection = false
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var trackerManager: TrackerManager? = null
    private val modelString = "yolov8n_float16.tflite"
    private val labelFilename = "file:///android_asset/coco.txt"
    private val inputSize = 640
    private val isV8 = true
    private val isQuantized = false
    private val numThreads = 1
    private val locationHandler = Handler()
    private var lastTrackerManager: TrackerManager? = null

    private val isDebug = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(null) // Changed
        Timber.d("onCreate $this")
        //initialize binding & UI
        binding = TfeOdActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        chronometer = binding.chronoContainer
        binding.wasteCounter.text = "0"
        bottomSheet = BottomSheet(binding)
        hideSystemUI()

        setupPermissions()

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        updateLocation()

        imageProcessor = ImageProcessor()
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
                this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                REQUEST_LOCATION_PERMISSION
            )
            return
        }
        val newLocation = fusedLocationClient.lastLocation

        newLocation.addOnSuccessListener {

            if (it != null) {
                bottomSheet.showGPSCoordinates(
                    arrayOf(
                        it.longitude.toString(),
                        it.latitude.toString()
                    )
                )
            }
            location = it

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

    private fun setupPermissions() {
        val permissions = arrayOf(
            PERMISSION_CAMERA,
            PERMISSION_LOCATION,
            Manifest.permission.ACCESS_FINE_LOCATION
        )
        if (checkPermissions(permissions)) {
            setFragment()
        } else {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(permissions, PERMISSIONS_REQUEST)
            }
        }
    }

    private fun checkPermissions(permissions: Array<String>): Boolean {
        for (permission in permissions) {
            if (ContextCompat.checkSelfPermission(this, permission)
                != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }


    /** Callback for Camera2 API  */
    override fun onImageAvailable(reader: ImageReader) {
        try {
            imageProcessor.openCameraImage(reader, previewWidth, previewHeight)
            if (detectorPaused) {
                imageProcessor.readyForNextImage()
            } else {
                processImage()
            }
        } catch (e: Exception) {
            Timber.e(e, "Exception!")
            return
        }
    }

    @Synchronized
    public override fun onStart() {
        Timber.d("onStart $this")
        super.onStart()
    }

    @Synchronized
    public override fun onResume() {
        Timber.d("onResume $this")
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

    @Synchronized
    public override fun onPause() {
        Timber.d("onPause $this")
        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            Timber.e(e, "Exception!")
        }
        super.onPause()
    }

    @Synchronized
    public override fun onStop() {
        Timber.d("onStop $this")
        super.onStop()
    }

    @Synchronized
    public override fun onDestroy() {
        Timber.d("onDestroy $this")
        super.onDestroy()
        locationHandler.removeCallbacksAndMessages(null)
    }

    @Synchronized
    private fun runInBackground(r: Runnable?) {
        if (handler != null) {
            handler!!.post(r!!)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSIONS_REQUEST && checkPermissions(permissions))
            setFragment()
    }

    private fun setFragment() {
        val cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        val cameraId = chooseCamera(cameraManager)
        val camera2Fragment = CameraConnectionFragment.newInstance(
            chronometer,
            { getCount() },
            { size: Size?, rotation: Int ->
                previewHeight = size!!.height
                previewWidth = size.width
                onPreviewSizeChosen(size, rotation)
            },
            { startDetector() }, { endDetector() }, this, desiredPreviewFrameSize,
        )

        camera2Fragment.setCamera(cameraId)
        supportFragmentManager.beginTransaction().replace(R.id.container, camera2Fragment).commit()
    }

    private val screenOrientation: Int
        get() = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }

    fun manageVisibility() {
        if (binding.chronometer.visibility == View.VISIBLE) {
            binding.chronometer.visibility = View.INVISIBLE
        } else {
            binding.chronometer.visibility = View.VISIBLE
        }
    }

    fun updateCounter(count: Int?) {
        binding.wasteCounter.text = count.toString()
        if (count != null) {
            wasteCount = count
        }
    }

    private fun getCount(): Int {
        return wasteCount
    }

    private fun endDetector() {
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

    private fun startDetector() {
        val context = this
        try {
            //create detector only one time
            if (detector == null) {
                detector = YoloDetector.create(
                    assets,
                    modelString,
                    labelFilename,
                    CONFIDENCE_THRESHOLD,
                    isQuantized,
                    isV8,
                    inputSize
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
        if (trackerManager != null) {
            bottomSheet.displayDetection(trackerManager!!)
        }
        val cropSize = detector?.inputSize
        cropSize?.let {
            Timber.i(Bitmap.createBitmap(it, it, Bitmap.Config.ARGB_8888).toString())
            croppedBitmap = Bitmap.createBitmap(it, it, Bitmap.Config.ARGB_8888)
            frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth,
                previewHeight,
                it,
                it,
                sensorOrientation,
                MAINTAIN_ASPECT
            )
        }

        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)
        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay?.addCallback(object : OverlayView.DrawCallback {
            override fun drawCallback(canvas: Canvas?) {
                if (canvas != null) {
                    trackerManager?.draw(canvas, context, previewWidth, previewHeight)
                }
                if (isDebug) {
                    if (canvas != null) {
                        trackerManager?.drawDebug(canvas)
                    }
                }
                trackerManager?.let { tracker ->
                    updateCounter(tracker.detectedWaste.size)
                }
                // ImageUtils.drawDebugScreen(canvas, previewWidth, previewHeight, cropToFrameTransform)
            }
        })
    }

    private fun onPreviewSizeChosen(size: Size?, rotation: Int?) {
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


    private fun processImage() {
        trackingOverlay?.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            imageProcessor.readyForNextImage()
            return
        }
        computingDetection = true

        imageProcessor.getRgbBytes()?.let {
            rgbFrameBitmap?.setPixels(
                it, 0, previewWidth, 0, 0, previewWidth, previewHeight
            )
        }
        imageProcessor.readyForNextImage()
        if (croppedBitmap != null && rgbFrameBitmap != null && frameToCropTransform != null) {
            val canvas = Canvas(croppedBitmap!!)
            canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)
            // For examining the actual TF input.
            if (SAVE_PREVIEW_BITMAP) {
                ImageUtils.saveBitmap(croppedBitmap!!)
            }
        }
        runInBackground {
            val startTime = SystemClock.uptimeMillis()
            val results: List<Detector.Recognition>? = croppedBitmap?.let {
                detector?.recognizeImage(croppedBitmap)
            }

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            val mappedRecognitions =
                ImageUtils.mapDetectionsWithTransform(results, cropToFrameTransform)
            trackerManager?.processDetections(mappedRecognitions, location)
            trackingOverlay?.postInvalidate()
            computingDetection = false
            runOnUiThread {
                bottomSheet.showInference(lastProcessingTimeMs.toString() + "ms")
            }
        }
    }

    private val desiredPreviewFrameSize: Size
        get() = Size(1280, 720)

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val MAINTAIN_ASPECT = true
        private const val SAVE_PREVIEW_BITMAP = false
        private const val REQUEST_LOCATION_PERMISSION = 2

        private const val PERMISSIONS_REQUEST = 1
        private const val PERMISSION_CAMERA = Manifest.permission.CAMERA
        private const val PERMISSION_LOCATION = Manifest.permission.ACCESS_FINE_LOCATION
    }
}