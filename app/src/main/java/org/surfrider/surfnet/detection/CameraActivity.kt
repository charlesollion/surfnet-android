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
import android.app.Fragment
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.hardware.Camera
import android.hardware.Camera.PreviewCallback
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image.Plane
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Trace
import android.util.Size
import android.view.Surface
import android.view.View
import android.view.ViewTreeObserver.OnGlobalLayoutListener
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetBehavior.BottomSheetCallback
import org.surfrider.surfnet.detection.databinding.TfeOdActivityCameraBinding
import org.surfrider.surfnet.detection.env.ImageUtils.convertYUV420SPToARGB8888
import org.surfrider.surfnet.detection.env.ImageUtils.convertYUV420ToARGB8888
import org.surfrider.surfnet.detection.env.Logger
import java.io.IOException

abstract class CameraActivity : AppCompatActivity(), OnImageAvailableListener, PreviewCallback,
    View.OnClickListener {

    lateinit var binding: TfeOdActivityCameraBinding

    @JvmField
    protected var previewWidth = 0

    @JvmField
    protected var previewHeight = 0
    val isDebug = false

    @JvmField
    protected var handler: Handler? = null
    private var handlerThread: HandlerThread? = null
    private var useCamera2API = false
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    private var luminanceStride = 0
        private set
    private var defaultModelIndex = 0
    private var defaultDeviceIndex = 0
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null

    @JvmField
    protected var modelStrings = ArrayList<String>()
    private var sheetBehavior: BottomSheetBehavior<LinearLayout?>? = null

    /** Current indices of device and model.  */
    @JvmField
    var currentDevice = -1

    @JvmField
    var currentModel = -1

    @JvmField
    var currentNumThreads = -1

    @JvmField
    var deviceStrings = ArrayList<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        LOGGER.d("onCreate $this")
        super.onCreate(null)
        //initialize binding
        binding = TfeOdActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setupToolbar()
        setupPermissions()
        setupBottomSheetLayout()
    }

    private fun setupBottomSheetLayout() {
        val bottomSheetLayout = binding.bottomSheetLayout
        deviceStrings.add("CPU")
        deviceStrings.add("GPU")
        deviceStrings.add("NNAPI")
        currentDevice = defaultDeviceIndex
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout.bottomSheetLayout)
        modelStrings = getModelStrings(assets, ASSET_PATH)
        currentModel = defaultModelIndex
        currentNumThreads =
            bottomSheetLayout.threads.text.toString().trim { it <= ' ' }.toInt()
        val deviceAdapter = ArrayAdapter(
            this@CameraActivity, R.layout.deviceview_row, R.id.deviceview_row_text, deviceStrings
        )
        val modelAdapter = ArrayAdapter(
            this@CameraActivity, R.layout.listview_row, R.id.listview_row_text, modelStrings
        )
        val vto = bottomSheetLayout.gestureLayout.viewTreeObserver
        vto.addOnGlobalLayoutListener(
            object : OnGlobalLayoutListener {
                override fun onGlobalLayout() {
                    bottomSheetLayout.gestureLayout.viewTreeObserver.removeOnGlobalLayoutListener(
                        this
                    )
                    val height = bottomSheetLayout.gestureLayout.measuredHeight
                    sheetBehavior!!.peekHeight = height
                }
            })
        sheetBehavior?.isHideable = false
        sheetBehavior?.addBottomSheetCallback(
            object : BottomSheetCallback() {
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

        with(bottomSheetLayout.deviceList) {
            choiceMode = ListView.CHOICE_MODE_SINGLE
            adapter = deviceAdapter
            setItemChecked(defaultDeviceIndex, true)
            onItemClickListener =
                AdapterView.OnItemClickListener { _, _, _, _ -> updateActiveModel() }
        }

        with(bottomSheetLayout.modelList) {
            choiceMode = ListView.CHOICE_MODE_SINGLE
            adapter = modelAdapter
            setItemChecked(defaultModelIndex, true)
            onItemClickListener =
                AdapterView.OnItemClickListener { _, _, _, _ -> updateActiveModel() }
        }
        bottomSheetLayout.plus.setOnClickListener(this)
        bottomSheetLayout.minus.setOnClickListener(this)
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
    }

    private fun setupPermissions() {
        if (hasPermission()) {
            setFragment()
        } else {
            requestPermission()
        }
    }

    private fun getModelStrings(mgr: AssetManager, path: String?): ArrayList<String> {
        val res = ArrayList<String>()
        try {
            val files = mgr.list(path!!)
            for (file in files!!) {
                val splits =
                    file.split("\\.".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
                if (splits[splits.size - 1] == "tflite") {
                    res.add(file)
                }
            }
        } catch (e: IOException) {
            System.err.println("getModelStrings: " + e.message)
        }
        return res
    }

    protected fun getRgbBytes(): IntArray? {
        imageConverter?.run()
        return rgbBytes
    }

    protected val luminance: ByteArray?
        get() = yuvBytes[0]

    /** Callback for android.hardware.Camera API  */
    override fun onPreviewFrame(bytes: ByteArray, camera: Camera) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!")
            return
        }
        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                val previewSize = camera.parameters.previewSize
                previewHeight = previewSize.height
                previewWidth = previewSize.width
                rgbBytes = IntArray(previewWidth * previewHeight)
                onPreviewSizeChosen(Size(previewSize.width, previewSize.height), 90)
            }
        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            return
        }
        isProcessingFrame = true
        yuvBytes[0] = bytes
        luminanceStride = previewWidth
        imageConverter =
            Runnable { convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes!!) }
        postInferenceCallback = Runnable {
            camera.addCallbackBuffer(bytes)
            isProcessingFrame = false
        }
        processImage()
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
            processImage()
        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            Trace.endSection()
            return
        }
        Trace.endSection()
    }

    @Synchronized
    public override fun onStart() {
        LOGGER.d("onStart $this")
        super.onStart()
    }

    @Synchronized
    public override fun onResume() {
        LOGGER.d("onResume $this")
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

    @Synchronized
    public override fun onPause() {
        LOGGER.d("onPause $this")
        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            LOGGER.e(e, "Exception!")
        }
        super.onPause()
    }

    @Synchronized
    public override fun onStop() {
        LOGGER.d("onStop $this")
        super.onStop()
    }

    @Synchronized
    public override fun onDestroy() {
        LOGGER.d("onDestroy $this")
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
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment()
            } else {
                requestPermission()
            }
        }
    }

    private fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                    this@CameraActivity,
                    "Camera permission is required for this demo",
                    Toast.LENGTH_LONG
                )
                    .show()
            }
            requestPermissions(arrayOf(PERMISSION_CAMERA), PERMISSIONS_REQUEST)
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private fun isHardwareLevelSupported(
        characteristics: CameraCharacteristics, requiredLevel: Int
    ): Boolean {
        val deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)!!
        return if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            requiredLevel == deviceLevel
        } else requiredLevel <= deviceLevel
        // deviceLevel is not LEGACY, can use numerical sort
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
                val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    ?: continue

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL
                        || isHardwareLevelSupported(
                    characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL
                ))
                LOGGER.i("Camera API lv2?: %s", useCamera2API)
                return cameraId
            }
        } catch (e: CameraAccessException) {
            LOGGER.e(e, "Not allowed to access camera")
        }
        return null
    }

    private fun setFragment() {
        val cameraId = chooseCamera()
        val fragment: Fragment
        if (useCamera2API) {
            val camera2Fragment = CameraConnectionFragment.newInstance(
                { size, rotation ->
                    previewHeight = size.height
                    previewWidth = size.width
                    onPreviewSizeChosen(size, rotation)
                },
                this,
                layoutId,
                desiredPreviewFrameSize
            )
            camera2Fragment.setCamera(cameraId)
            fragment = camera2Fragment
        } else {
            fragment = LegacyCameraConnectionFragment(this, layoutId, desiredPreviewFrameSize)
        }
        fragmentManager.beginTransaction().replace(R.id.container, fragment).commit()
    }

    private fun fillBytes(planes: Array<Plane>, yuvBytes: Array<ByteArray?>) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity())
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]]
        }
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

    //  @Override
    //  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    //    setUseNNAPI(isChecked);
    //    if (isChecked) apiSwitchCompat.setText("NNAPI");
    //    else apiSwitchCompat.setText("TFLITE");
    //  }
    override fun onClick(v: View) {
        if (v.id == R.id.plus) {
            val threads = binding.bottomSheetLayout.threads.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            if (numThreads >= 9) return
            numThreads++
            binding.bottomSheetLayout.threads.text = numThreads.toString()
            setNumThreads(numThreads)
        } else if (v.id == R.id.minus) {
            val threads = binding.bottomSheetLayout.threads.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            if (numThreads == 1) {
                return
            }
            numThreads--
            binding.bottomSheetLayout.threads.text = numThreads.toString()
            setNumThreads(numThreads)
        }
    }

    protected fun showFrameInfo(frameInfo: String?) {
        binding.bottomSheetLayout.frameInfo.text = frameInfo
    }

    protected fun showCropInfo(cropInfo: String?) {
        binding.bottomSheetLayout.cropInfo.text = cropInfo
    }

    protected fun showInference(inferenceTime: String?) {
        binding.bottomSheetLayout.inferenceInfo!!.text = inferenceTime
    }

    protected abstract fun updateActiveModel()
    protected abstract fun processImage()
    protected abstract fun onPreviewSizeChosen(size: Size?, rotation: Int?)
    protected abstract val layoutId: Int
    protected abstract val desiredPreviewFrameSize: Size?
    protected abstract fun setNumThreads(numThreads: Int)
    protected abstract fun setUseNNAPI(isChecked: Boolean)

    companion object {
        private val LOGGER = Logger()
        private const val PERMISSIONS_REQUEST = 1
        private const val PERMISSION_CAMERA = Manifest.permission.CAMERA
        private const val ASSET_PATH = ""
        private fun allPermissionsGranted(grantResults: IntArray): Boolean {
            for (result in grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    return false
                }
            }
            return true
        }
    }
}