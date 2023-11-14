package org.surfrider.surfnet.detection.env

import android.graphics.Bitmap
import android.media.ImageReader
import android.os.Trace
import org.opencv.android.Utils
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
                Timber.i("ImageConverter: ${Thread.currentThread().name}")
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
        val rgbFrameBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
        getRgbBytes()?.let {
            val downsampledRGBInts = downsampleRGBInts(it, previewWidth, previewHeight, downsampleFactor)
            rgbFrameBitmap.setPixels(
                downsampledRGBInts, 0, newWidth, 0, 0, newWidth, newHeight
            )
        }
        readyForNextImage()
        val bmp32: Bitmap = rgbFrameBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val currFrame = Mat()
        Utils.bitmapToMat(bmp32, currFrame)
        return currFrame
    }
}