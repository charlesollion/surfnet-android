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


import android.content.pm.PackageManager
import android.location.LocationManager
import android.graphics.*
import android.media.ImageReader.OnImageAvailableListener
import android.os.Handler
import android.os.SystemClock
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.surfrider.surfnet.detection.customview.OverlayView
import org.surfrider.surfnet.detection.customview.OverlayView.DrawCallback
import org.surfrider.surfnet.detection.env.BorderedText
import org.surfrider.surfnet.detection.env.ImageUtils.getTransformationMatrix
import org.surfrider.surfnet.detection.env.ImageUtils.saveBitmap
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import org.surfrider.surfnet.detection.tflite.YoloDetector
import org.surfrider.surfnet.detection.tracking.MultiBoxTracker
import timber.log.Timber
import java.io.IOException
import java.util.LinkedList
import android.Manifest
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationRequest
import android.os.Build
import android.os.Bundle
import androidx.annotation.RequiresApi
import java.util.*

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
open class DetectorActivity : CameraActivity(), OnImageAvailableListener, LocationListener {
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
    private var tracker: MultiBoxTracker? = null
    private var borderedText: BorderedText? = null
    private var coordinates: Array<String> = arrayOf("", "")
    private lateinit var locationManager: LocationManager
    private val modelString = "yolov8n_float16.tflite"
    private val labelFilename = "file:///android_asset/coco.txt"
    private val inputSize = 640
    private val isV8 = true
    private val isQuantized = false
    private val numThreads = 1
    public override fun onPreviewSizeChosen(size: Size?, rotation: Int?) {
        val textSizePx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics
        )
        borderedText = BorderedText(textSizePx)
        borderedText?.setTypeface(Typeface.MONOSPACE)
        tracker = MultiBoxTracker(this)
        try {
            detector = YoloDetector.create(
                assets, modelString, labelFilename, isQuantized, isV8, inputSize
            )
            detector?.useGpu()
            detector?.setNumThreads(numThreads)
        } catch (e: IOException) {
            e.printStackTrace()
            Timber.e(e, "Exception initializing Detector!")
            val toast = Toast.makeText(
                applicationContext, "Detector could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
        val cropSize = detector?.inputSize
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
        cropSize?.let {
            croppedBitmap = Bitmap.createBitmap(it, it, Bitmap.Config.ARGB_8888)
            frameToCropTransform = getTransformationMatrix(
                previewWidth, previewHeight,
                it, it,
                sensorOrientation, MAINTAIN_ASPECT
            )
        }
        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)
        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay?.addCallback(
            object : DrawCallback {
                override fun drawCallback(canvas: Canvas?) {
                    tracker?.draw(canvas)
                    if (isDebug) {
                        tracker?.drawDebug(canvas)
                    }
                }
            })
        tracker?.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation)
    }

    private fun getLocation() {
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if ((ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED)) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 2)
        }
        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000, 5f, this)
    }
    override fun onLocationChanged(location: Location) {
        coordinates[0] = location.latitude.toString()
        coordinates[1] = location.longitude.toString()
    }
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 2) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show()
            }
            else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }



    private fun getGPSInfo() {
        val REQUEST_LOCATION_PERMISSION = 2
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), REQUEST_LOCATION_PERMISSION)
        } else {
            getLocation()
        }
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
        Timber.i("Preparing image $currTimestamp for detection in bg thread.")
        rgbFrameBitmap?.setPixels(
            getRgbBytes(),
            0,
            previewWidth,
            0,
            0,
            previewWidth,
            previewHeight
        )
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
            Timber.i("Running detection on image $currTimestamp")
            val startTime = SystemClock.uptimeMillis()
            val results: List<Recognition>? = detector?.recognizeImage(croppedBitmap)
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
            tracker?.trackResults(mappedRecognitions, currTimestamp)
            trackingOverlay?.postInvalidate()
            computingDetection = false
            getGPSInfo()
            runOnUiThread {
                showFrameInfo(previewWidth.toString() + "x" + previewHeight)
                showCropInfo(
                    cropCopyBitmap?.width.toString() + "x" + cropCopyBitmap?.height
                )
                showInference(lastProcessingTimeMs.toString() + "ms")
                showGPSCoordinates(coordinates)
            }
        }
    }

    override val layoutId: Int
        get() = R.layout.tfe_od_camera_connection_fragment_tracking
    override val desiredPreviewFrameSize: Size?
        get() = Size(640, 640)

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
    }
}