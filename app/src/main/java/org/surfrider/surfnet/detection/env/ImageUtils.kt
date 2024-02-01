/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.surfrider.surfnet.detection.env

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.media.Image
import android.os.Environment
import org.surfrider.surfnet.detection.tflite.Detector
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.util.ArrayList
import java.util.LinkedList
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** Utility class for manipulating images.  */
object ImageUtils {
    // This value is 2 ^ 18 - 1, and is used to clamp the RGB values before their ranges
    // are normalized to eight bits.
    private const val kMaxChannelValue = 262143

    /**
     * Saves a Bitmap object to disk for analysis.
     *
     * @param bitmap The bitmap to save.
     */
    @JvmStatic
    fun saveBitmap(bitmap: Bitmap, context:Context) {
        val filename = "preview.png"
        val root =
            File(context.getExternalFilesDir(null), "tensorflow")
        Timber.i("Saving %dx%d bitmap to %s.", bitmap.width, bitmap.height, root)
        if (!root.mkdirs()) {
            Timber.i("Make dir failed")
        }
        val file = File(root, filename)
        if (file.exists()) {
            file.delete()
        }
        try {
            val out = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.PNG, 99, out)
            out.flush()
            out.close()
        } catch (e: Exception) {
            Timber.e(e, "Exception!")
        }
    }

    public fun rgbIntToByteArray(rgbInts: IntArray): ByteArray {
        val outputBytes = ByteArray(rgbInts.size * 3)
        for(i: Int in rgbInts.indices) {
            val rgb = rgbInts[i]
            outputBytes[i*3] = ((rgb shr 16) and 0xff).toByte()
            outputBytes[i*3+1] = ((rgb shr 8) and 0xff).toByte()
            outputBytes[i*3+2] = (rgb and 0xff).toByte()
        }
        return outputBytes
    }

    @JvmStatic
    private fun mapYUVtoRGB(valY: Int, valU: Int, valV: Int): Int {
        // Adjust and check YUV values
        var y = valY
        var u = valU
        var v = valV
        y = if (y - 16 < 0) 0 else y - 16
        u -= 128
        v -= 128

        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);
        val y1192 = 1192 * y
        var r = y1192 + 1634 * v
        var g = y1192 - 833 * v - 400 * u
        var b = y1192 + 2066 * u

        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        r = if (r > kMaxChannelValue) kMaxChannelValue else if (r < 0) 0 else r
        g = if (g > kMaxChannelValue) kMaxChannelValue else if (g < 0) 0 else g
        b = if (b > kMaxChannelValue) kMaxChannelValue else if (b < 0) 0 else b
        return -0x1000000 or (r shl 6 and 0xff0000) or (g shr 2 and 0xff00) or (b shr 10 and 0xff)
    }

    @JvmStatic
    fun convertYUV420ToARGB8888(
        yData: ByteArray,
        uData: ByteArray,
        vData: ByteArray,
        width: Int,
        height: Int,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        out: IntArray
    ) {
        var yp = 0
        for (j in 0 until height) {
            val pY = yRowStride * j
            val pUV = uvRowStride * (j shr 1)
            for (i in 0 until width) {
                val uvOffset = pUV + (i shr 1) * uvPixelStride
                out[yp++] = mapYUVtoRGB(
                    0xff and yData[pY + i].toInt(),
                    0xff and uData[uvOffset].toInt(),
                    0xff and vData[uvOffset].toInt()
                )
            }
        }
    }

    /**
     * Returns a transformation matrix from one reference frame into another. Handles cropping (if
     * maintaining aspect ratio is desired) and rotation.
     *
     * @param srcWidth Width of source frame.
     * @param srcHeight Height of source frame.
     * @param dstWidth Width of destination frame.
     * @param dstHeight Height of destination frame.
     * @param applyRotation Amount of rotation to apply from one frame to another. Must be a multiple
     * of 90.
     * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
     * cropping the image if necessary.
     * @return The transformation fulfilling the desired requirements.
     */
    @JvmStatic
    fun getTransformationMatrix(
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int,
        applyRotation: Int,
        maintainAspectRatio: Boolean
    ): Matrix {
        val matrix = Matrix()
        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                Timber.w("Rotation of %d % 90 != 0", applyRotation)
            }

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)

            // Rotate around origin.
            matrix.postRotate(applyRotation.toFloat())
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        val transpose = (abs(applyRotation) + 90) % 180 == 0
        val inWidth = if (transpose) srcHeight else srcWidth
        val inHeight = if (transpose) srcWidth else srcHeight
        Timber.i("--------------- inWidth = ${inWidth}, dstWidth $dstWidth")
        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            val scaleFactorX = dstWidth / inWidth.toFloat()
            val scaleFactorY = dstHeight / inHeight.toFloat()
            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                // Crop should be centered so we apply translations
                matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)
                val scaleFactor = max(scaleFactorX, scaleFactorY)
                Timber.i("----------------- scaleFactor = ${scaleFactor}, scaleFactorX $scaleFactorX scaleFactorY $scaleFactorY")
                matrix.postScale(scaleFactor, scaleFactor)
                matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY)
            }
        }
        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
        }
        return matrix
    }

    @JvmStatic
    fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                Timber.d("Initializing buffer %d at size %d", i, buffer.capacity())
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]!!]
        }
    }

    @JvmStatic
    fun downsampleRGBInts(rgbInts: IntArray, originalWidth: Int, originalHeight: Int, downsampleFactor: Int): IntArray {
        if (downsampleFactor == 1) {
            return rgbInts
        }
        val newWidth = originalWidth / downsampleFactor
        val newHeight = originalHeight / downsampleFactor
        val outputInts = IntArray(newWidth * newHeight)
        for(i: Int in 0..<newHeight) {
            for (j: Int in 0..<newWidth) {
                outputInts[i*newWidth+j] = rgbInts[i*originalWidth+j*downsampleFactor]
            }
        }
        return outputInts
    }
    fun drawBorder(canvas: Canvas, previewWidth: Int, previewHeight: Int) {
        // Debug function to show frame size and crop size
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 10.0f
        // Slightly smaller than Camera frame width to see all borders
        val rectCam = RectF(5.0F, 5.0F, previewWidth.toFloat()-5.0F, previewHeight.toFloat()-5.0F)

        val frameToCanvasTransform = Matrix()
        val scale = min(canvas.width / previewWidth.toFloat(), canvas.height / previewHeight.toFloat())
        frameToCanvasTransform.postScale(scale, scale)

        // Draw Camera frame
        frameToCanvasTransform.mapRect(rectCam)
        canvas.drawRect(rectCam, paint)
    }


    fun drawDetections(canvas: Canvas, results: List<Detector.Recognition>?, previewWidth: Int, previewHeight: Int) {
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        val frameToCanvasTransform = Matrix()
        val scale = min(
            canvas.width / previewWidth.toFloat(), canvas.height / previewHeight.toFloat()
        )
        frameToCanvasTransform.postScale(scale, scale)
        if (results != null) {
            for (result in results) {
                val location = result.location
                if (location != null) {
                    frameToCanvasTransform.mapRect(location)
                    canvas.drawRect(location, paint)
                }
            }
        }
    }

    fun drawOFLinesPRK(canvas: Canvas, outputLinesFlow: ArrayList<FloatArray>, previewWidth: Int, previewHeight: Int) {
        val paint = Paint()
        paint.color = Color.WHITE
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4.0f
        // Timber.i("output line flow size: ${outputLinesFlow.size}")
        val frameToCanvasTransform = Matrix()
        val scale = min(canvas.width / previewWidth.toFloat(),
            canvas.height / previewHeight.toFloat())
        frameToCanvasTransform.postScale(scale, scale)
        for(line in outputLinesFlow) {
            val points = line.clone()
            frameToCanvasTransform.mapPoints(points)
            //Timber.i(" flow - i, j, dx, dy, $i, $j, $dx, $dy")
            canvas.drawCircle(points[0], points[1], 10.0f, paint)
            canvas.drawLine(points[0], points[1], points[2], points[3], paint)
        }
    }


    fun mapDetectionsWithTransform(results: List<Detector.Recognition>?, cropToFrameTransform: Matrix?): MutableList<Detector.Recognition> {
        val mappedRecognitions: MutableList<Detector.Recognition> = LinkedList()
        if (results != null) {
            for (result in results) {
                val location = result.location
                if (location != null) {
                    cropToFrameTransform?.mapRect(location)
                    result.location = location
                    mappedRecognitions.add(result)
                }
            }
        }
        return mappedRecognitions
    }
}