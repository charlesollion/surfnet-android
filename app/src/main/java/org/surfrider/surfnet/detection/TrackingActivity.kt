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
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.TableRow
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.Granularity
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.sync.Mutex
import org.opencv.android.CameraActivity
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.surfrider.surfnet.detection.customview.BottomSheet
import org.surfrider.surfnet.detection.customview.OverlayView
import org.surfrider.surfnet.detection.databinding.ActivityCameraBinding
import org.surfrider.surfnet.detection.env.ImageProcessor
import org.surfrider.surfnet.detection.env.ImageUtils
import org.surfrider.surfnet.detection.env.ImageUtils.drawDetections
import org.surfrider.surfnet.detection.env.ImageUtils.drawOFLinesPRK
import org.surfrider.surfnet.detection.flow.DenseOpticalFlow
import org.surfrider.surfnet.detection.flow.IMU_estimator
import org.surfrider.surfnet.detection.tflite.Detector
import org.surfrider.surfnet.detection.tflite.YoloDetector
import org.surfrider.surfnet.detection.tracking.TrackerManager
import timber.log.Timber
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.max


@OptIn(DelicateCoroutinesApi::class)
class TrackingActivity : CameraActivity(), CvCameraViewListener2, LocationListener {

    private lateinit var locationManager: LocationManager
    private lateinit var binding: ActivityCameraBinding
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var bottomSheet: BottomSheet
    private lateinit var chronoContainer: TableRow
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var outputLinesFlow: ArrayList<FloatArray>
    private lateinit var imuEstimator: IMU_estimator
    private lateinit var opticalFlow: DenseOpticalFlow
    private lateinit var openCvCameraView: CameraBridgeViewBase
    private lateinit var frameRgba: Mat
    private lateinit var frameResized: Mat
    private lateinit var cropRect: Rect
    private lateinit var frameGray: Mat

    private var previewWidth = 0
    private var previewHeight = 0
    private var resizedWidth = 0.0
    private var resizedHeight = 0.0
    private var detectorPaused = true
    private var avgFlowSpeed: PointF? = null
    private var wasteCount = 0
    private var location: Location? = null
    private var fastSelfMotionTimestamp: Long = 0

    @OptIn(ExperimentalCoroutinesApi::class)
    private val threadDetector = newSingleThreadContext("InferenceThread")

    private val backgroundScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int = 0
    private var detector: YoloDetector? = null
    private var lastProcessingTimeMs: Long = 0
    private var croppedBitmap: Bitmap? = null
    private var computingDetection = false
    private var computingOF = false
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var trackerManager: TrackerManager? = null
    private var latestDetections: List<Detector.Recognition>? = null
    private val mutex = Mutex()
    private val locationHandler = Handler(Looper.getMainLooper())
    private var lastTrackerManager: TrackerManager? = null
    private val isDebug = false
    private var isGpsActivate = false

    private var lastPause: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Timber.d("onCreate $this")

        if (OpenCVLoader.initLocal()) Timber.i("Successful opencv loading")

        //initialize binding & UI
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
        openCvCameraView = findViewById(R.id.camera_view) as CameraBridgeViewBase
        openCvCameraView.visibility = CameraBridgeViewBase.VISIBLE
        openCvCameraView.setCvCameraViewListener(this)

        trackerManager = TrackerManager()

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        chronoContainer = binding.chronoContainer
        binding.wasteCounter.text = "0"
        bottomSheet = BottomSheet(binding)
        hideSystemUI()

        setupPermissions()

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)


        val mLocationRequest: LocationRequest =
            LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000).apply {
                setMinUpdateDistanceMeters(1F)
                setGranularity(Granularity.GRANULARITY_PERMISSION_LEVEL)
                setWaitForAccurateLocation(true)
            }.build()
        val mLocationCallback: LocationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult) {
                location = locationResult.lastLocation
            }
        }
        fusedLocationClient.requestLocationUpdates(mLocationRequest, mLocationCallback, null)
        updateLocation()
        imageProcessor = ImageProcessor()

        val mLocationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        isGpsActivate = mLocationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
        if (!isGpsActivate) {
            val locationPermissionDialog = LocationPermissionDialog()
            //TODO change supportFragmentManager
            // locationPermissionDialog.show(supportFragmentManager, "stop_record_dialog")
        }

        // init IMU_estimator, optical flow
        imuEstimator = IMU_estimator(this.applicationContext)
        opticalFlow = DenseOpticalFlow()
        outputLinesFlow = arrayListOf()

        binding.closeButton.setOnClickListener {
            Timber.i("close button")
            val intent = Intent(applicationContext, TutorialActivity::class.java)
            startActivity(intent)
        }

        binding.startButton.setOnClickListener {
            Timber.i("start button")
            startDetector()
        }

        binding.stopButton.setOnClickListener {
            Timber.i("stop button")
            endDetector()
        }
    }

    override fun onPause() {
        super.onPause()
        if (openCvCameraView != null) openCvCameraView.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (openCvCameraView != null) openCvCameraView.enableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (openCvCameraView != null) openCvCameraView.disableView()
        locationHandler.removeCallbacksAndMessages(null)
    }

    override fun getCameraViewList(): List<CameraBridgeViewBase?>? {
        return listOf<CameraBridgeViewBase>(openCvCameraView)
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Timber.i("CameraViewStarted ${width}x${height}")
        previewWidth = width
        previewHeight = height

        /*if (openCvCameraView.rotation != null) {
                    sensorOrientation = openCvCameraView.rotation - screenOrientation
                }
        Timber.i("Camera orientation relative to screen canvas: %d", sensorOrientation)*/

        // Get Android Graphics transform matrix
        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight, INPUT_SIZE, INPUT_SIZE, sensorOrientation, MAINTAIN_ASPECT
        )

        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)

        // Manual transforms for CVMats: first resize, then crop
        // TODO merge both android graphics and opencv transforms into one single function
        val scaleFactorX = INPUT_SIZE.toDouble() / previewWidth
        val scaleFactorY = INPUT_SIZE.toDouble() / previewHeight
        val scaleFactor = max(scaleFactorX, scaleFactorY)
        resizedWidth = scaleFactor * previewWidth
        resizedHeight = scaleFactor * previewHeight

        cropRect = Rect((resizedWidth / 2 - INPUT_SIZE/2).toInt(), (resizedHeight/2- INPUT_SIZE/2).toInt(), INPUT_SIZE, INPUT_SIZE)

        // Initialize CVMat frames
        frameRgba = Mat(height, width, CvType.CV_8UC4)
        frameResized = Mat(resizedHeight.toInt(), resizedWidth.toInt(), CvType.CV_8UC3)
        frameGray = Mat(height, width, CvType.CV_8UC1)

        // Initialize Bitmap as input to detector
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    }

    override fun onCameraViewStopped() {
        frameRgba.release()
        frameResized.release()
        frameGray.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        // device acceleration
        computeIMU()

        // optical flow
        inputFrame?.let {
            outputLinesFlow = opticalFlow.run(it.gray(), 1)
            if (fastSelfMotionTimestamp == 0L) {
                avgFlowSpeed = trackerManager?.associateFlowWithTrackers(
                    outputLinesFlow,
                    FLOW_REFRESH_RATE_MILLIS
                )
            }
        }

        // detection (only if not already detecting)
        if(!detectorPaused) {
            if (!computingDetection) {
                detect(frameRgba)
            }
        }

        // display
        frameRgba = inputFrame!!.rgba()
        trackingOverlay?.let {
            it.postInvalidate()
        }
        return frameRgba
    }

    private fun endDetector() {
        //adapt screen
        binding.startButton.visibility = View.VISIBLE
        binding.stopButton.visibility = View.INVISIBLE
        binding.redLine.visibility = View.VISIBLE
        val stopRecordDialog = trackerManager?.let { StopRecordDialog(it) }
        //TODO change supportFragmentManager
        // stopRecordDialog?.show(supportFragmentManager, "stop_record_dialog")

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
        detectorPaused = false



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
                        Timber.i("Fast motion! canvas: ${canvas.width}x${canvas.height} - preview:${previewWidth}x${previewHeight}")
                        ImageUtils.drawBorder(it, previewWidth, previewHeight)
                    }
                    ImageUtils.drawCrop(it, previewWidth, previewHeight, INPUT_SIZE, cropToFrameTransform!!)
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

    @Suppress("DEPRECATION")
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
            trackerManager?.let {
                val date = Calendar.getInstance().time
                val iso8601Format =
                    SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.getDefault())
                val iso8601DateString = iso8601Format.format(date)
                it.addPosition(location = location, date = iso8601DateString)
            }
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
        if (!checkPermissions(permissions)) {
            // TODO add dialog for permissions
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

    fun updateCounter(count: Int?) {
        binding.wasteCounter.text = count.toString()
        if (count != null) {
            wasteCount = count
        }
    }

    private fun computeIMU() {
        // get IMU variables
        val velocity: FloatArray = imuEstimator.velocity
        val imuPosition: FloatArray = imuEstimator.position
        // Convert the velocity to kmh
        val xVelocity = velocity[0] * 3.6f
        val yVelocity = velocity[1] * 3.6f
        val zVelocity = velocity[2] * 3.6f

        // Get the magnitude of the velocity vector
        val speed =
            kotlin.math.sqrt((xVelocity * xVelocity + yVelocity * yVelocity + zVelocity * zVelocity).toDouble())
                .toFloat()

        if (speed > MAX_SELF_VELOCITY) {
            fastSelfMotionTimestamp = System.currentTimeMillis()
        } else {
            // If the time since last timestamp is over VELOCITY_PAUSE_STICKINESS, we reset the timestamp
            if (System.currentTimeMillis() - fastSelfMotionTimestamp > VELOCITY_PAUSE_STICKINESS) {
                fastSelfMotionTimestamp = 0
            }
        }
        runOnUiThread {
            bottomSheet.showIMUStats(
                applicationContext,
                arrayOf(
                    imuPosition[0], imuPosition[1], imuPosition[2],
                    speed, avgFlowSpeed?.x ?: 0.0F, avgFlowSpeed?.y ?: 0.0F
                )
            )
        }
    }

    private fun detect(frame: Mat) {
        computingDetection = true
        Imgproc.resize(frame, frameResized, Size(resizedWidth, resizedHeight))
        Timber.i("input frame: ${frame.size()} frame after resize: ${frameResized.size()} rect: ${cropRect}")
        val rgbaInnerWindow = frameResized.submat(cropRect)
        Utils.matToBitmap(rgbaInnerWindow, croppedBitmap);
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap!!, applicationContext)
        }
        /*trackerManager?.let {
            it.updateTrackers()
        }*/

        // run in background
        backgroundScope.launch(threadDetector) {
            val startTime = SystemClock.uptimeMillis()
            val results: List<Detector.Recognition>? = croppedBitmap?.let {
                detector?.recognizeImage(it)
            }
            @Synchronized
            latestDetections = results

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            val mappedRecognitions =
                ImageUtils.mapDetectionsWithTransform(results, cropToFrameTransform)
            /*mutex.withLock {
                if (fastSelfMotionTimestamp == 0L) {
                    trackerManager?.processDetections(mappedRecognitions, location)
                    trackerManager?.updateTrackers()
                }
            }*/

            trackingOverlay?.postInvalidate()
            computingDetection = false
            runOnUiThread {
                bottomSheet.showInference(lastProcessingTimeMs.toString() + "ms")
            }
        }
    }

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