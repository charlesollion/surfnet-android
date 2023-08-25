package org.surfrider.surfnet.detection

import android.annotation.SuppressLint
import android.app.Fragment
import android.graphics.SurfaceTexture
import android.hardware.Camera
import android.hardware.Camera.CameraInfo
import android.hardware.Camera.PreviewCallback
import android.os.Bundle
import android.os.HandlerThread
import android.util.Size
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.TextureView.SurfaceTextureListener
import android.view.View
import android.view.ViewGroup
import org.surfrider.surfnet.detection.customview.AutoFitTextureView
import org.surfrider.surfnet.detection.env.ImageUtils.getYUVByteSize
import org.surfrider.surfnet.detection.env.Logger
import java.io.IOException

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
 */@SuppressLint("ValidFragment")
class LegacyCameraConnectionFragment(
    private val imageListener: PreviewCallback,
    /** The layout identifier to inflate for this Fragment.  */
    private val layout: Int, private val desiredSize: Size?
) : Fragment() {
    private var camera: Camera? = null

    /** An [AutoFitTextureView] for camera preview.  */
    private var textureView: AutoFitTextureView? = null

    /**
     * [TextureView.SurfaceTextureListener] handles several lifecycle events on a [ ].
     */
    private val surfaceTextureListener: SurfaceTextureListener = object : SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(
            texture: SurfaceTexture, width: Int, height: Int
        ) {
            val index: Int = cameraId
            camera = Camera.open(index)
            camera?.let {
                try {
                    val parameters = it.getParameters()
                    val focusModes = parameters.supportedFocusModes
                    if (focusModes != null
                        && focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)
                    ) {
                        parameters.focusMode = Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE
                    }
                    val cameraSizes = parameters.supportedPreviewSizes
                    val sizes = arrayOfNulls<Size>(cameraSizes.size)
                    var i = 0
                    for (size in cameraSizes) {
                        sizes[i++] = Size(size.width, size.height)
                    }
                    val previewSize = desiredSize?.let { it1 ->
                        CameraConnectionFragment.chooseOptimalSize(
                            sizes, it1.width, desiredSize.height
                        )
                    }
                    if (previewSize != null) {
                        parameters.setPreviewSize(previewSize.width, previewSize.height)
                    }
                    it.setDisplayOrientation(90)
                    it.parameters = parameters
                    it.setPreviewTexture(texture)
                } catch (exception: IOException) {
                    it.release()
                }
                it.setPreviewCallbackWithBuffer(imageListener)
                val s = it.getParameters().previewSize
                it.addCallbackBuffer(ByteArray(getYUVByteSize(s.height, s.width)))
                textureView?.setAspectRatio(s.height, s.width)
                it.startPreview()
            }
        }

        override fun onSurfaceTextureSizeChanged(
            texture: SurfaceTexture, width: Int, height: Int
        ) {
        }

        override fun onSurfaceTextureDestroyed(texture: SurfaceTexture): Boolean {
            return true
        }

        override fun onSurfaceTextureUpdated(texture: SurfaceTexture) {}
    }

    /** An additional thread for running tasks that shouldn't block the UI.  */
    private var backgroundThread: HandlerThread? = null
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle
    ): View? {
        return inflater.inflate(layout, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        textureView = view.findViewById<View>(R.id.texture) as AutoFitTextureView
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
        // When the screen is turned off and turned back on, the SurfaceTexture is already
        // available, and "onSurfaceTextureAvailable" will not be called. In that case, we can open
        // a camera and start preview from here (otherwise, we wait until the surface is ready in
        // the SurfaceTextureListener).
        if (textureView?.isAvailable == true) {
            camera?.startPreview()
        } else {
            textureView?.surfaceTextureListener = surfaceTextureListener
        }
    }

    override fun onPause() {
        stopCamera()
        stopBackgroundThread()
        super.onPause()
    }

    /** Starts a background thread and its [Handler].  */
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground")
        backgroundThread?.start()
    }

    /** Stops the background thread and its [Handler].  */
    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
        } catch (e: InterruptedException) {
            LOGGER.e(e, "Exception!")
        }
    }

    protected fun stopCamera() {
        if (camera != null) {
            camera?.stopPreview()
            camera?.setPreviewCallback(null)
            camera?.release()
            camera = null
        }
    }

    private val cameraId: Int
        private get() {
            val ci = CameraInfo()
            for (i in 0 until Camera.getNumberOfCameras()) {
                Camera.getCameraInfo(i, ci)
                if (ci.facing == CameraInfo.CAMERA_FACING_BACK) return i
            }
            return -1 // No camera found
        }

    companion object {
        private val LOGGER = Logger()

        /** Conversion from screen rotation to JPEG orientation.  */
        private val ORIENTATIONS = SparseIntArray()

        init {
            ORIENTATIONS.append(Surface.ROTATION_0, 90)
            ORIENTATIONS.append(Surface.ROTATION_90, 0)
            ORIENTATIONS.append(Surface.ROTATION_180, 270)
            ORIENTATIONS.append(Surface.ROTATION_270, 180)
        }
    }
}