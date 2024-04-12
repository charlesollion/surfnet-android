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
import android.graphics.Bitmap.createScaledBitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import org.opencv.core.Mat
import org.surfrider.surfnet.detection.env.MathUtils.sigmoid
import org.surfrider.surfnet.detection.tflite.Detector
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.util.ArrayList
import kotlin.math.abs
import kotlin.math.max

/** Utility class for manipulating images.  */
object ImageUtils {
    private var alreadySavedThisSession = false

    /**
     * Saves a Bitmap object to disk for analysis.
     *
     * @param bitmap The bitmap to save.
     */
    @JvmStatic
    fun saveBitmap(bitmap: Bitmap, context:Context) {
        if(alreadySavedThisSession)
            return
        val filename = "input_preview.png"
        val root =
            File(context.getExternalFilesDir(null), "tensorflow")
        Timber.i("Saving %dx%d bitmap to %s.", bitmap.width, bitmap.height, root)
        if (root.mkdirs()) {
            Timber.i("created folder $root")
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
    fun loadBitmap(context: Context): Bitmap? {
        val filename = "input_preview.png"
        val root = File(context.getExternalFilesDir(null), "tensorflow")
        val file = File(root, filename)

        if (!file.exists()) {
            Timber.e("File does not exist: $file")
            return null
        }

        return try {
            BitmapFactory.decodeFile(file.absolutePath)
        } catch (e: Exception) {
            Timber.e(e, "Exception while loading bitmap!")
            null
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
                Timber.i("--------------- scaleFactor = ${scaleFactor}, scaleFactorX $scaleFactorX scaleFactorY $scaleFactorY")
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

    fun drawCrop(canvas: Canvas, frameToCanvasTransform:Matrix, cropSize: Int, cropToFrameTransform:Matrix) {
        // Debug function to show crop size
        val paint = Paint()
        paint.color = Color.GREEN
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 10.0f
        // Crop size
        val rectCrop = RectF(0.0F, 0.0F, cropSize * 1.0F, cropSize * 1.0F)

        cropToFrameTransform.mapRect(rectCrop)
        frameToCanvasTransform.mapRect(rectCrop)
        canvas.drawRect(rectCrop, paint)
    }

    fun drawBorder(canvas: Canvas, frameToCanvasTransform: Matrix, previewWidth: Int, previewHeight: Int) {
        // Draw borders of screen
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 10.0f
        // Slightly smaller than Camera frame width to see all borders
        val rectCam = RectF(5.0F, 5.0F, previewWidth.toFloat()-5.0F, previewHeight.toFloat()-5.0F)

        // Draw Camera frame
        frameToCanvasTransform.mapRect(rectCam)
        canvas.drawRect(rectCam, paint)
    }

    fun drawDetections(canvas: Canvas, results: List<Detector.Recognition>?, frameToCanvasTransform: Matrix, isMovedDetections:Boolean, drawOnlyMasks:Boolean) {
        val paint = Paint()
        if (isMovedDetections)
            paint.color = Color.BLUE
        else
            paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        if (results != null) {
            for (result in results) {
                val newLocation = RectF(result.location)
                newLocation?.let { location ->
                    frameToCanvasTransform.mapRect(location)
                    if(!drawOnlyMasks)
                        canvas.drawRect(location, paint)
                    // Draw mask
                    if(isMovedDetections) {
                        result.bitmap?.let { bitmap ->
                            canvas.drawBitmap(bitmap, null, location, null)
                        }
                    }
                }
            }
        }
    }

    fun drawOFLinesPRK(canvas: Canvas, outputLinesFlow: ArrayList<FloatArray>, frameToCanvasTransform: Matrix) {
        val paint = Paint()
        paint.color = Color.WHITE
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4.0f
        for(line in outputLinesFlow) {
            val points = line.clone()
            frameToCanvasTransform.mapPoints(points)
            canvas.drawCircle(points[0], points[1], 10.0f, paint)
            canvas.drawLine(points[0], points[1], points[2], points[3], paint)
        }
    }

    private fun buildBitmapFromMask(mask: Mat?, location: RectF, rgb: IntArray): Bitmap? {
        mask?.let {
            val w = location.width().toInt()
            val h = location.height().toInt()
            if (w < 4 || h < 4) {
                Timber.i("Warning too mask small region ${w}x$h")
                return null
            }
            val outputBitmap = Bitmap.createBitmap(
                w / 4,
                h / 4,
                Bitmap.Config.ARGB_8888
            )
            for (i in 0 until outputBitmap.width)  {
                for (j in 0 until outputBitmap.height) {

                    var pixelValue = 0
                    if (sigmoid(it.get(j, i)[0]) > 0.5)
                        pixelValue = 128
                    outputBitmap.setPixel(i, j, Color.argb(pixelValue, rgb[0], rgb[1], rgb[2]))
                }
            }
            return createScaledBitmap(outputBitmap, w, h, true)
        }
        return null
    }

    private fun getRGB(i: Int): IntArray {
        when(i%6) {
            0 -> return intArrayOf(200, 128, 0)
            1 -> return intArrayOf(128, 200, 0)
            2 -> return intArrayOf(0, 128, 200)
            3 -> return intArrayOf(0, 200, 128)
            4 -> return intArrayOf(128, 0, 200)
            5 -> return intArrayOf(200, 0, 128)
        }
        return intArrayOf(200, 128, 0)
    }

    fun mapDetectionsWithTransform(results: List<Detector.Recognition>?, cropToFrameTransform: Matrix?) {
        // Performs inplace mapping of detections
        if (results != null) {
            for (result in results) {
                result.bitmap = result.mask?.let { mask ->
                    buildBitmapFromMask(mask, result.location, getRGB(result.detectedClass))
                }
                val newLocation = RectF(result.location)
                newLocation?.let { location ->
                    cropToFrameTransform?.mapRect(location)
                }
                result.location = newLocation
            }
        }
    }
}