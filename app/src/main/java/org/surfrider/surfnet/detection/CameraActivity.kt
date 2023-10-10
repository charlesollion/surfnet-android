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
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.location.LocationManager
import android.media.Image.Plane
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.*
import android.util.Size
import android.view.Surface
import android.view.View
import android.view.ViewTreeObserver.OnGlobalLayoutListener
import android.view.WindowManager
import android.widget.Chronometer
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetBehavior.BottomSheetCallback
import kotlinx.coroutines.Dispatchers
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import org.opencv.core.CvType.*
import org.opencv.core.Mat
import org.opencv.core.*
import org.surfrider.surfnet.detection.databinding.TfeOdActivityCameraBinding
import org.surfrider.surfnet.detection.env.ImageUtils.convertYUV420ToARGB8888
import org.surfrider.surfnet.detection.flow.DenseOpticalFlow
import org.surfrider.surfnet.detection.flow.IMU_estimator
import timber.log.Timber
import java.text.DecimalFormat
import java.util.*


abstract class CameraActivity : AppCompatActivity(), OnImageAvailableListener {


    private lateinit var locationManager: LocationManager
    private lateinit var binding: TfeOdActivityCameraBinding


    @JvmField
    protected var previewWidth = 0

    @JvmField
    protected var previewHeight = 0
    val isDebug = false

    @JvmField
    protected var handler: Handler? = null
    private var handlerThread: HandlerThread? = null
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    private var luminanceStride = 0
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null
    private lateinit var df : DecimalFormat;
    private lateinit var imuEstimator : IMU_estimator
    private lateinit var opticalFlow : DenseOpticalFlow
    public lateinit var outputFlow : Mat
    public lateinit var outputLinesFlow: ArrayList<FloatArray>

    private var sheetBehavior: BottomSheetBehavior<LinearLayout?>? = null
    var detectorPaused = true
    lateinit var chronometer: Chronometer

    override fun onCreate(savedInstanceState: Bundle?) {
        Timber.d("onCreate $this")
        super.onCreate(null)
        //initialize binding
        binding = TfeOdActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        df = DecimalFormat("#.##")
        // init IMU_estimator, optical flow
        imuEstimator = IMU_estimator(this.applicationContext)
        opticalFlow = DenseOpticalFlow()
        outputFlow = Mat()
        outputLinesFlow = arrayListOf()

        chronometer = binding.chronometer

        setupPermissions()
        setupBottomSheetLayout()

        lifecycleScope.launch(Dispatchers.Default) {
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                while(true) {
                    scheduledOpticalFlow()
                    delay(250)
                }
            }
        }
    }
    private fun setupBottomSheetLayout() {
        val bottomSheetLayout = binding.bottomSheetLayout
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout.bottomSheetLayout)

        val vto = bottomSheetLayout.gestureLayout.viewTreeObserver
        vto.addOnGlobalLayoutListener(object : OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                bottomSheetLayout.gestureLayout.viewTreeObserver.removeOnGlobalLayoutListener(
                    this
                )
                val height = bottomSheetLayout.gestureLayout.measuredHeight
                sheetBehavior!!.peekHeight = height
            }
        })
        sheetBehavior?.isHideable = false
        sheetBehavior?.addBottomSheetCallback(object : BottomSheetCallback() {
            override fun onStateChanged(bottomSheet: View, newState: Int) {
                when (newState) {
                    BottomSheetBehavior.STATE_HIDDEN -> {}
                    BottomSheetBehavior.STATE_EXPANDED -> {
                        bottomSheetLayout.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_down)
                    }

                    BottomSheetBehavior.STATE_COLLAPSED -> {
                        bottomSheetLayout.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_up)
                    }

                    BottomSheetBehavior.STATE_DRAGGING -> {}
                    BottomSheetBehavior.STATE_SETTLING -> bottomSheetLayout.bottomSheetArrow.setImageResource(
                        R.drawable.icn_chevron_up
                    )
                }
            }
            override fun onSlide(bottomSheet: View, slideOffset: Float) {}
        })
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

    protected fun getRgbBytes(): IntArray? {
        imageConverter?.run()
        return rgbBytes
    }

    /** Callback for Camera2 API  */
    override fun onImageAvailable(reader: ImageReader) {



        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return
        }
        if (rgbBytes == null) {
            rgbBytes = IntArray(previewWidth * previewHeight)
        }
        try {
            val image = reader.acquireLatestImage() ?: return
            if (isProcessingFrame) {
                image.close()
                return
            }
            isProcessingFrame = true

            // Processing frame
            Trace.beginSection("imageAvailable")
            val planes = image.planes
            fillBytes(planes, yuvBytes)
            luminanceStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            imageConverter = Runnable {
                convertYUV420ToARGB8888(
                    yuvBytes[0]!!,
                    yuvBytes[1]!!,
                    yuvBytes[2]!!,
                    previewWidth,
                    previewHeight,
                    luminanceStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes!!
                )
            }
            postInferenceCallback = Runnable {
                image.close()
                isProcessingFrame = false
            }

            if(detectorPaused) {
                readyForNextImage()
            } else {
                processImage()
            }
        } catch (e: Exception) {
            Timber.e(e, "Exception!")
            Trace.endSection()
            return
        }
        Trace.endSection()

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
    }

    @Synchronized
    protected fun runInBackground(r: Runnable?) {
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


    private fun chooseCamera(): String? {
        val manager = getSystemService(CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)

                // We don't use a front facing camera in this sample.
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue
                }
                characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    ?: continue

                return cameraId
            }
        } catch (e: CameraAccessException) {
            Timber.e(e, "Not allowed to access camera")
        }
        return null
    }

    private fun setFragment() {
        val cameraId = chooseCamera()
        val camera2Fragment = desiredPreviewFrameSize?.let {
            CameraConnectionFragment.newInstance(
                    chronometer,
                { size: Size?, rotation: Int ->
                    previewHeight = size!!.height
                    previewWidth = size.width
                    onPreviewSizeChosen(size, rotation)
                }, { startDetector() }, { endDetector() }, this, it,
            )
        }
        camera2Fragment?.setCamera(cameraId)

        if (camera2Fragment != null) {
            supportFragmentManager.beginTransaction().replace(R.id.container, camera2Fragment)
                .commit()
        }
    }

    private fun fillBytes(planes: Array<Plane>, yuvBytes: Array<ByteArray?>) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                Timber.d("Initializing buffer %d at size %d", i, buffer.capacity())
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]]
        }
    }

    private suspend fun scheduledOpticalFlow() {
        Timber.d("##### background run flow and IMU")
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


        if(rgbBytes == null) {
            return
        }
        val rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        getRgbBytes()?.let {
            rgbFrameBitmap.setPixels(
                it, 0, previewWidth, 0, 0, previewWidth, previewHeight
            )
        }
        val bmp32: Bitmap = rgbFrameBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val currFrame = Mat()
        Utils.bitmapToMat(bmp32, currFrame)

        // outputFlow = opticalFlow.run(currFrame)
        outputLinesFlow = opticalFlow.run(currFrame)

        // Timber.i("### flow output: " + df.format(outputFlow.x) + " / " + df.format((outputFlow.y)))
        // outputFlow.x.toFloat(), outputFlow.y.toFloat()
        showIMUStats(arrayOf(imuPosition[0], imuPosition[1], imuPosition[2], speed, 0.0F, 0.0F))
    }

    protected fun readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback!!.run()
        }
    }

    protected val screenOrientation: Int
        get() = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }


    protected fun showInference(inferenceTime: String?) {
        binding.bottomSheetLayout.inferenceInfo.text = inferenceTime
    }

    protected fun showGPSCoordinates(coordinates: Array<String>?) {
        if (coordinates != null && coordinates.size == 2) {
            binding.bottomSheetLayout.latitudeInfo.text = coordinates[0]
            binding.bottomSheetLayout.longitudeInfo.text = coordinates[1]
        } else {
            binding.bottomSheetLayout.latitudeInfo.text = "null"
            binding.bottomSheetLayout.longitudeInfo.text = "null"
        }
    }

    protected fun showIMUStats(stats: Array<Float>?) {
        if (stats != null && stats.size == 6) {
            binding.bottomSheetLayout.positionInfo.text = df.format(stats[0]) + " " +  df.format(stats[1]) + " " +  df.format(stats[2])
            binding.bottomSheetLayout.speedInfo.text = df.format(stats[3])
            binding.bottomSheetLayout.flowInfo.text = df.format(stats[4]) + " " + df.format(stats[5])
        } else {
            binding.bottomSheetLayout.positionInfo.text = "null"
            binding.bottomSheetLayout.speedInfo.text = "null"
            binding.bottomSheetLayout.flowInfo.text = "null"
        }
    }

    protected abstract fun processImage()
    protected abstract fun onPreviewSizeChosen(size: Size?, rotation: Int?)
    protected abstract fun startDetector()
    protected abstract fun endDetector()
    protected abstract val layoutId: Int
    protected abstract val desiredPreviewFrameSize: Size?

    companion object {
        private const val PERMISSIONS_REQUEST = 1
        private const val PERMISSION_CAMERA = Manifest.permission.CAMERA
        private const val PERMISSION_LOCATION = Manifest.permission.ACCESS_FINE_LOCATION
    }
}