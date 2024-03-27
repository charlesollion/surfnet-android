package org.surfrider.surfnet.detection.tflite

import android.graphics.RectF
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.surfrider.surfnet.detection.env.MathUtils.sigmoid
import timber.log.Timber
import java.util.PriorityQueue

object DetectorUtils {
    //non maximum suppression
    fun nms(list: ArrayList<Detector.Recognition>, numClass: Int): ArrayList<Detector.Recognition?> {
        val nmsList = ArrayList<Detector.Recognition?>()
        for (k in 0 until numClass) {
            //1.find max confidence per class
            val pq = PriorityQueue<Detector.Recognition?>(
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
                val max = detections[0]
                nmsList.add(max)
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection = detections[j]
                    val b = detection!!.location
                    if (boxIou(max!!.location, b) < mNmsThresh) {
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

    fun weightedSumOfMasks(maskMatrix: Mat, maskWeights: FloatArray, maskResolutionH: Int, rect: Rect): Mat {
        // Expects a maskMatrix of size (numMasks, maskResolutionW * maskResolutionH)

        val numMasks = maskWeights.size
        val weightMatrix = Mat(1, numMasks, CvType.CV_32F)
        weightMatrix.put(0, 0, maskWeights)
        /*for (i in maskWeights.indices) {
            var value: Double = maskWeights[i].toDouble()
            //var value = 0.0
            if (i>1) {
                value = 0.0
            }
            weightMatrix.put(0, i, value)
        }*/


        // Calculate the weighted sum of masks
        val weightedSum = Mat(1, maskResolutionH * maskResolutionH, CvType.CV_32F)
        Core.gemm(weightMatrix, maskMatrix, 1.0, Mat(), 0.0, weightedSum)
        val fullMask = weightedSum.reshape(1, maskResolutionH)

        // val finalOutput = Mat()
        // Core.compare(Mat(fullMask, rect), Scalar(MASK_THRESHOLD), finalOutput, Core.CMP_GE)

        return Mat(fullMask, rect)
    }

    private const val mNmsThresh = 0.6f
    private const val MASK_THRESHOLD = 0.0
}