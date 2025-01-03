package org.surfrider.surfnet.detection.tflite

import android.content.Context
import android.content.res.AssetManager
import android.graphics.RectF
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import timber.log.Timber
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.PrintWriter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.PriorityQueue

object DetectorUtils {
    @JvmStatic
    @Throws(IOException::class)
    fun loadModelFile(assets: AssetManager, modelFilename: String?): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename!!)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    //non maximum suppression
    fun nms(list: ArrayList<Detector.Recognition>, numClass: Int): ArrayList<Detector.Recognition> {
        val nmsList = ArrayList<Detector.Recognition>()
        for (k in 0 until numClass) {
            //1.find max confidence per class
            val pq = PriorityQueue<Detector.Recognition>(
                50
            ) { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                rhs.confidence.compareTo(lhs.confidence)
            }
            for (i in list.indices) {
                if (list[i].detectedClass == k) {
                    pq.add(list[i])
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a = arrayOfNulls<Detector.Recognition>(pq.size)
                val detections = pq.toArray(a)
                val max = detections[0]!!
                nmsList.add(max)
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection = detections[j]
                    val b = detection!!.location
                    if (boxIou(max.location, b) < mNmsThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsList
    }
    // non max suppression returning indices only (for segmentation)
    fun nmsIndices(list: ArrayList<Detector.Recognition>, numClass: Int): ArrayList<Int> {
        val nmsList = ArrayList<Int>()
        for (k in 0 until numClass) {
            //1.find max confidence per class
            val pq = PriorityQueue<Int>(
                50
            ) { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                list[rhs].confidence.compareTo(list[lhs].confidence)
            }
            for (i in list.indices) {
                if (list[i].detectedClass == k) {
                    pq.add(i)
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a = arrayOfNulls<Int>(pq.size)
                val detections = pq.toArray(a)
                val max = detections[0]!!
                nmsList.add(max)

                pq.clear()
                for (j in 1 until detections.size) {
                    val detection = detections[j]!!
                    val b = list[detection].location
                    if (boxIou(list[max].location, b) < mNmsThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsList
    }

    private fun boxIou(a: RectF, b: RectF): Float {
        return boxIntersection(a, b) / boxUnion(a, b)
    }

    private fun boxIntersection(a: RectF, b: RectF): Float {
        val w = overlap(
            (a.left + a.right) / 2, a.right - a.left,
            (b.left + b.right) / 2, b.right - b.left
        )
        val h = overlap(
            (a.top + a.bottom) / 2, a.bottom - a.top,
            (b.top + b.bottom) / 2, b.bottom - b.top
        )
        return if (w < 0 || h < 0) 0.0F else w * h
    }

    private fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }
    private fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = if (l1 > l2) l1 else l2
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = if (r1 < r2) r1 else r2
        return right - left
    }
    fun saveMatAsText(mat: Mat, filename: String, context: Context) {
        Timber.i("saving $filename, context: $context")
        // Ensure that the Mat is continuous
        val continuousMat = Mat()
        mat.copyTo(continuousMat)

        // Open a PrintWriter to write to the file
        val root =
            File(context.getExternalFilesDir(null), "tensorflow")
        val file = File(root, filename)
        if (file.exists()) {
            Timber.i("text file already exists: $file")
            return
        }
        val writer = PrintWriter(file)

        // Write each element of the Mat to the file
        for (i in 0 until continuousMat.rows()) {
            for (j in 0 until continuousMat.cols()) {
                val value = continuousMat.get(i, j)[0]
                writer.print("$value ")
            }
            writer.println()
        }

        // Close the PrintWriter
        writer.close()
    }

    fun weightedSumOfMasks(maskMatrix: Mat, maskWeights: Mat, maskResolutionH: Int, rect: Rect): Mat {
        // Expects a maskMatrix of size (numMasks, maskResolutionW * maskResolutionH)

        // Calculate the weighted sum of masks
        val weightedSum = Mat(1, maskResolutionH * maskResolutionH, CvType.CV_32F)
        Core.gemm(maskWeights, maskMatrix, 1.0, Mat(), 0.0, weightedSum)
        val fullMask = weightedSum.reshape(1, maskResolutionH)

        return Mat(fullMask, rect)
    }

    private const val mNmsThresh = 0.6f
}