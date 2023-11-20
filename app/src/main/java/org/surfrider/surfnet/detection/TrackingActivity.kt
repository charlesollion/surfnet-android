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
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.PointF
import android.hardware.camera2.CameraManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Build
import android.os.Bundle
import android.os.Handler
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
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.surfrider.surfnet.detection.customview.BottomSheet
import org.surfrider.surfnet.detection.customview.OverlayView
import org.surfrider.surfnet.detection.databinding.FragmentSendDataDialogBinding
import org.surfrider.surfnet.detection.databinding.ActivityCameraBinding
import org.surfrider.surfnet.detection.env.ImageProcessor
import org.surfrider.surfnet.detection.env.ImageUtils
import org.surfrider.surfnet.detection.env.ImageUtils.drawDetections
import org.surfrider.surfnet.detection.env.ImageUtils.drawOFLinesPRK
import org.surfrider.surfnet.detection.env.Utils.chooseCamera
import org.surfrider.surfnet.detection.flow.DenseOpticalFlow
import org.surfrider.surfnet.detection.flow.IMU_estimator
import org.surfrider.surfnet.detection.tflite.Detector
import org.surfrider.surfnet.detection.tflite.YoloDetector
import org.surfrider.surfnet.detection.tracking.TrackerManager
import timber.log.Timber
import java.io.IOException
import java.util.*


@OptIn(DelicateCoroutinesApi::class)
class TrackingActivity : AppCompatActivity(), OnImageAvailableListener, LocationListener {

    private lateinit var locationManager: LocationManager
    private lateinit var binding: ActivityCameraBinding
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var bottomSheet: BottomSheet
    private lateinit var chronoContainer: TableRow
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var outputLinesFlow: ArrayList<FloatArray>
    private lateinit var imuEstimator: IMU_estimator
    private lateinit var opticalFlow: DenseOpticalFlow

    private var previewWidth = 0

    private var previewHeight = 0
    private var detectorPaused = true
    private var flowRegionUpdateNeeded = false
    private var wasteCount = 0
    private var location: Location? = null
    private var fastSelfMotionTimestamp: Long = 0

    private val threadDetector = newSingleThreadContext("InferenceThread")
    private val threadOpticalFlow = newSingleThreadContext("OpticalFlowThread")

    private var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int = 0
    private var detector: YoloDetector? = null
    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var currFrameMat: Mat? = null
    private var computingDetection = false
    private var computingOF = false
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var trackerManager: TrackerManager? = null
    private var latestDetections: List<Detector.Recognition>? = null
    private var currROIs: Mat? = null
    private val mutex = Mutex()
    private val locationHandler = Handler()
    private var lastTrackerManager: TrackerManager? = null
    private lateinit var bindingDialog: FragmentSendDataDialogBinding
    private val isDebug = false
    private var isGpsActivate = false

    private var lastPause: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(null)
        Timber.d("onCreate $this")

        if (OpenCVLoader.initDebug()) Timber.i("Successful opencv loading")

        //initialize binding & UI
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
        bindingDialog = FragmentSendDataDialogBinding.inflate(
            layoutInflater
        )
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        chronoContainer = binding.chronoContainer
        binding.wasteCounter.text = "0"
        bottomSheet = BottomSheet(binding)
        hideSystemUI()

        setupPermissions()

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        updateLocation()
        imageProcessor = ImageProcessor()

        val mLocationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        isGpsActivate = mLocationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
        if (!isGpsActivate) {
            val locationPermissionDialog = LocationPermissionDialog()
            locationPermissionDialog.show(supportFragmentManager, "stop_record_dialog")
        }

        // init IMU_estimator, optical flow
        imuEstimator = IMU_estimator(this.applicationContext)
        opticalFlow = DenseOpticalFlow()
        outputLinesFlow = arrayListOf()

        binding.closeButton.setOnClickListener {
            val intent = Intent(applicationContext, TutorialActivity::class.java)
            startActivity(intent)
        }

        binding.startButton.setOnClickListener {
            startDetector()
        }

        binding.stopButton.setOnClickListener {
            endDetector()
        }

    }

    private fun endDetector() {
        binding.startButton.visibility = View.VISIBLE
        binding.stopButton.visibility = View.INVISIBLE
        binding.redLine.visibility = View.VISIBLE
        trackerManager?.let { tracker ->
            lastTrackerManager = tracker
        }
        detectorPaused = true
        //removes all drawings of the trackingOverlay from the screen
        trackingOverlay?.invalidate()

        //reset trackers
        croppedBitmap = null
        cropToFrameTransform = null
        trackerManager = null
        trackingOverlay = null

        lastPause = SystemClock.elapsedRealtime()
        binding.chronometer.stop()
        val stopRecordDialog = StopRecordDialog(wasteCount, 2F)
        stopRecordDialog.show(supportFragmentManager, "stop_record_dialog")
    }

    private fun startDetector() {
        with(binding) {
            startButton.visibility = View.INVISIBLE
            stopButton.visibility = View.VISIBLE
            chronoContainer.visibility = View.VISIBLE
            closeButton.visibility = View.INVISIBLE
            redLine.visibility = View.INVISIBLE
        }

        val context = this
        try {
            //create detector only one time
            detector = detector ?: YoloDetector.create(
                assets,
                MODEL_STRING,
                LABEL_FILENAME,
                CONFIDENCE_THRESHOLD,
                IS_QUANTIZED,
                IS_V8,
                INPUT_SIZE
            )
            detector?.useGpu()
            detector?.setNumThreads(NUM_THREADS)
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
        trackerManager = lastTrackerManager ?: TrackerManager()

        bottomSheet.displayDetection(trackerManager!!)

        detector?.inputSize?.let {
            croppedBitmap = Bitmap.createBitmap(it, it, Bitmap.Config.ARGB_8888)
            frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight, it, it, sensorOrientation, MAINTAIN_ASPECT
            )
        }

        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)

        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay?.addCallback(object : OverlayView.DrawCallback {
            override fun drawCallback(canvas: Canvas?) {
                canvas?.let {
                    trackerManager?.draw(
                        it, context, previewWidth, previewHeight, bottomSheet.showOF
                    )
                    if (bottomSheet.showOF) {
                        drawOFLinesPRK(it, outputLinesFlow, previewWidth, previewHeight)
                    }
                    if (bottomSheet.showBoxes) {
                        drawDetections(it, latestDetections, previewWidth, previewHeight)
                    }
                    if (isDebug) {
                        trackerManager?.drawDebug(it)
                    }
                    if (fastSelfMotionTimestamp > 0) {
                        ImageUtils.drawBorder(canvas, previewWidth, previewHeight)
                    }
                    // ImageUtils.drawDebugScreen(canvas, previewWidth, previewHeight, cropToFrameTransform)
                }
                trackerManager?.let { tracker ->
                    updateCounter(tracker.detectedWaste.size)
                }
            }
        })

        binding.chronometer.base = if (lastPause == 0L) {
            SystemClock.elapsedRealtime()
        } else {
            binding.chronometer.base + (SystemClock.elapsedRealtime() - lastPause)
        }
        binding.chronometer.start()
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
            window.decorView.systemUiVisibility =
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        }
    }

    @RequiresApi(Build.VERSION_CODES.R)
    private fun hideNavigationBar() {
        val windowInsetsController = WindowCompat.getInsetsController(window, window.decorView)
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
                this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), REQUEST_LOCATION_PERMISSION
            )
            return
        }
        val newLocation = fusedLocationClient.lastLocation

        newLocation.addOnSuccessListener {

            if (it != null) {
                bottomSheet.showGPSCoordinates(
                    arrayOf(
                        it.longitude.toString(), it.latitude.toString()
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
            PERMISSION_CAMERA, PERMISSION_LOCATION, Manifest.permission.ACCESS_FINE_LOCATION
        )
        if (checkPermissions(permissions)) {
            setFragment()
        } else {
            //val locationPermissionDialog = LocationPermissionDialog()
            //locationPermissionDialog.show(supportFragmentManager, "stop_record_dialog")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(permissions, PERMISSIONS_REQUEST)
            }
        }
    }

    private fun checkPermissions(permissions: Array<String>): Boolean {
        for (permission in permissions) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    permission
                ) != PackageManager.PERMISSION_GRANTED
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
            Timber.e(e, "Exception in ImageListener!")
            return
        }
    }

    @Synchronized
    public override fun onDestroy() {
        Timber.d("onDestroy $this")
        super.onDestroy()
        locationHandler.removeCallbacksAndMessages(null)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSIONS_REQUEST && checkPermissions(permissions)) setFragment()
    }

    private fun setFragment() {
        val cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        val cameraId = chooseCamera(cameraManager)
        val camera2Fragment = CameraConnectionFragment.newInstance(
            { size: Size?, rotation: Int ->
                previewHeight = size!!.height
                previewWidth = size.width
                onPreviewSizeChosen(size, rotation)
            },
            this, desiredPreviewFrameSize,
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
    fun updateCounter(count: Int?) {
        binding.wasteCounter.text = count.toString()
        if (count != null) {
            wasteCount = count
        }
    }

    private fun computeOF() {
        computingOF = true
        // get IMU variables
        val velocity: FloatArray = imuEstimator.velocity
        val imuPosition: FloatArray = imuEstimator.position
        // Convert the velocity to kmh
        val xVelocity = velocity[0] * 3.6f
        val yVelocity = velocity[1] * 3.6f
        val zVelocity = velocity[2] * 3.6f
        var avgFlowSpeed: PointF? = null

        // Get the magnitude of the velocity vector
        val speed =
            kotlin.math.sqrt((xVelocity * xVelocity + yVelocity * yVelocity + zVelocity * zVelocity).toDouble())
                .toFloat()

        if(speed > MAX_SELF_VELOCITY) {
            fastSelfMotionTimestamp = System.currentTimeMillis()
        } else {
            // If the time since last timestamp is over VELOCITY_PAUSE_STICKINESS, we reset the timestamp
            if(System.currentTimeMillis() - fastSelfMotionTimestamp > VELOCITY_PAUSE_STICKINESS) {
                fastSelfMotionTimestamp = 0
            }
        }


        currFrameMat = imageProcessor.getMatFromRGB(
            previewWidth,
            previewHeight,
            DOWNSAMPLING_FACTOR_FLOW
        )

        lifecycleScope.launch(threadOpticalFlow) {
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                mutex.withLock {
                    if (fastSelfMotionTimestamp == 0L) {
                        avgFlowSpeed = trackerManager?.associateFlowWithTrackers(
                            outputLinesFlow,
                            FLOW_REFRESH_RATE_MILLIS
                        )
                        if (flowRegionUpdateNeeded) {
                            flowRegionUpdateNeeded = false
                            currROIs =
                                trackerManager?.getCurrentRois(
                                    1280,
                                    720,
                                    DOWNSAMPLING_FACTOR_FLOW,
                                    60
                                )
                        }
                    }
                }

                currFrameMat?.let {
                    outputLinesFlow = opticalFlow.run(it, currROIs, DOWNSAMPLING_FACTOR_FLOW)
                }
                computingOF = false

                runOnUiThread {
                    bottomSheet.showIMUStats(
                        arrayOf(
                            imuPosition[0], imuPosition[1], imuPosition[2],
                            speed, avgFlowSpeed?.x ?: 0.0F, avgFlowSpeed?.y ?: 0.0F
                        )
                    )
                }
            }
        }
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
        if (computingDetection && computingOF) {
            imageProcessor.readyForNextImage()
            return
        }
        if(!computingOF) {
            // will run its own thread
            computeOF()
        }
        if(!computingDetection) {
            // will run its own thread
            detect()
        }

    }

    private fun detect() {
        computingDetection = true

        imageProcessor.getRgbBytes()?.let {
            rgbFrameBitmap?.setPixels(
                it, 0, previewWidth, 0, 0, previewWidth, previewHeight
            )
        }
        imageProcessor.readyForNextImage()

        /*trackerManager?.let {
            it.updateTrackers()
        }*/

        if (croppedBitmap != null && rgbFrameBitmap != null && frameToCropTransform != null) {
            val canvas = Canvas(croppedBitmap!!)
            canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)
            // For examining the actual TF input.
            if (SAVE_PREVIEW_BITMAP) {
                ImageUtils.saveBitmap(croppedBitmap!!)
            }
        }

        lifecycleScope.launch(threadDetector) {
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                val startTime = SystemClock.uptimeMillis()
                val results: List<Detector.Recognition>? = croppedBitmap?.let {
                    detector?.recognizeImage(croppedBitmap)
                }
                latestDetections = results

                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                val mappedRecognitions =
                    ImageUtils.mapDetectionsWithTransform(results, cropToFrameTransform)
                mutex.withLock {
                    if(fastSelfMotionTimestamp == 0L) {
                        trackerManager?.processDetections(mappedRecognitions, location)
                        trackerManager?.updateTrackers()
                    }
                }
                flowRegionUpdateNeeded = true

                trackingOverlay?.postInvalidate()
                computingDetection = false
                runOnUiThread {
                    bottomSheet.showInference(lastProcessingTimeMs.toString() + "ms")
                }
            }
        }
    }

    private val desiredPreviewFrameSize: Size
        get() = Size(1280, 720)

    companion object {
        private const val FLOW_REFRESH_RATE_MILLIS: Long = 50
        private const val DOWNSAMPLING_FACTOR_FLOW: Int = 2
        private const val MAX_SELF_VELOCITY = 5.0
        private const val VELOCITY_PAUSE_STICKINESS: Long = 500

        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val MODEL_STRING = "yolov8n_float16.tflite"
        private const val LABEL_FILENAME = "file:///android_asset/coco.txt"
        private const val INPUT_SIZE = 640
        private const val IS_V8 = true
        private const val IS_QUANTIZED = false
        private const val NUM_THREADS = 1

        private const val MAINTAIN_ASPECT = true
        private const val SAVE_PREVIEW_BITMAP = false
        private const val REQUEST_LOCATION_PERMISSION = 2
        private const val PERMISSIONS_REQUEST = 1
        private const val PERMISSION_CAMERA = Manifest.permission.CAMERA
        private const val PERMISSION_LOCATION = Manifest.permission.ACCESS_FINE_LOCATION
    }
}