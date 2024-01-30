package org.surfrider.surfnet.detection.env

import android.R.attr.height
import android.R.attr.width
import android.graphics.Bitmap
import android.media.Image
import android.media.ImageReader
import android.os.Trace
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.surfrider.surfnet.detection.env.ImageUtils.downsampleRGBInts
import timber.log.Timber


class ImageProcessor {
    @JvmField
    var rgbInts: IntArray? = null

    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var luminanceStride = 0
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null

    fun getMatFromCamera(image: Image, previewWidth: Int, previewHeight: Int, downsampleFactor: Int): Mat? {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return null
        }
        if (rgbInts == null) {
            rgbInts = IntArray(previewWidth * previewHeight)
        }
        try {
            val planes = image.planes
            ImageUtils.fillBytes(planes, yuvBytes)
            luminanceStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            ImageUtils.convertYUV420ToARGB8888(
                yuvBytes[0]!!,
                yuvBytes[1]!!,
                yuvBytes[2]!!,
                previewWidth,
                previewHeight,
                luminanceStride,
                uvRowStride,
                uvPixelStride,
                rgbInts!!
            )
            image.close()
            return getMatFromRGB(previewWidth, previewHeight, downsampleFactor)

        } catch (e: Exception) {
            Timber.e(e, "Exception!")
            return null
        }
    }

    fun openCameraImage(reader: ImageReader, previewWidth: Int, previewHeight: Int) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return
        }
        if (rgbInts == null) {
            rgbInts = IntArray(previewWidth * previewHeight)
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
            ImageUtils.fillBytes(planes, yuvBytes)
            luminanceStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            imageConverter = Runnable {
                ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0]!!,
                    yuvBytes[1]!!,
                    yuvBytes[2]!!,
                    previewWidth,
                    previewHeight,
                    luminanceStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbInts!!
                )
            }
            postInferenceCallback = Runnable {
                image.close()
                isProcessingFrame = false
            }
        } catch (e: Exception) {
            Timber.e(e, "Exception!")
            Trace.endSection()
            return
        }
        Trace.endSection()
    }
    fun getRgbBytes(): IntArray? {
        imageConverter?.run()
        return rgbInts
    }

    fun readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback!!.run()
        }
    }
    fun getMatFromRGB(previewWidth: Int, previewHeight: Int, downsampleFactor: Int): Mat? {
        if(rgbInts == null) {
            return null
        }
        val newWidth = previewWidth / downsampleFactor
        val newHeight = previewHeight / downsampleFactor

        val mat = Mat(newHeight, newWidth, CvType.CV_8UC3)

        getRgbBytes()?.let {
            val downsampledRGBInts = downsampleRGBInts(it, previewWidth, previewHeight, downsampleFactor)
            val bytes = ImageUtils.rgbIntToByteArray(downsampledRGBInts)
            mat.put(0,0, bytes)
        }
        // readyForNextImage()

        return mat
    }
}