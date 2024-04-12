package org.surfrider.surfnet.detection.flow

import android.graphics.PointF
import android.graphics.RectF
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import org.surfrider.surfnet.detection.TrackingActivity
import org.surfrider.surfnet.detection.tflite.Detector
import org.surfrider.surfnet.detection.tracking.TrackerManager
import timber.log.Timber
import java.util.LinkedList

public class OpticalFlow {
    private var prevGreyImage = Mat()
    private var prevPts = MatOfPoint2f()
    private var currPts = MatOfPoint2f()
    private val maxCorners = 50
    private val minDistance = 80.0
    private var flowPtsCount = 50
    private var status = MatOfByte()
    private var err = MatOfFloat()

    var outputLinesFlow: ArrayList<FloatArray> = arrayListOf()
    private var outputFlowLinesRollingArray = LinkedList<TrackingActivity.TimedOutputFlowLine>()

    private fun updatePoints(greyImage: Mat, scalingFactor: Int) {
        val corners = MatOfPoint()
        var numberCorners = maxCorners
        if(!prevPts.empty())
            numberCorners -= prevPts.rows()
        Imgproc.goodFeaturesToTrack(greyImage, corners, numberCorners, 0.1, 5.0)

        val newCornersArray = corners.toArray()

        // Combine the new points with the existing ones
        val combinedCorners = if (!prevPts.empty()) {
            val prevCorners = prevPts.toList()
            newCornersArray.toList() + prevCorners
        } else {
            newCornersArray.toList()
        }

        // Ensure we don't exceed maxCorners points
        val updatedCorners = if (combinedCorners.size > maxCorners) {
            combinedCorners.take(maxCorners)
        } else {
            combinedCorners
        }

        // Update prevPts with the updated points
        prevPts.fromList(updatedCorners)
    }


    fun run(newFrame: Mat, scalingFactor: Int) {
        outputLinesFlow = pyrLK(newFrame, scalingFactor)


    }

    fun updateRollingArray(currTime: Long) {
        outputFlowLinesRollingArray.add(
            TrackingActivity.TimedOutputFlowLine(
                outputLinesFlow,
                currTime
            )
        )
        if (outputFlowLinesRollingArray.size > FLOW_ROLLING_ARRAY_MAX_SIZE) {
            outputFlowLinesRollingArray.removeFirst()
        }
    }

    private fun pyrLK(currGreyImage: Mat, scalingFactor: Int) : ArrayList<FloatArray> {
        // if this is the first loop, find good features
        if (prevGreyImage.empty()) {
            currGreyImage.copyTo(prevGreyImage)
            updatePoints(currGreyImage, scalingFactor)
            return arrayListOf()
        } else {
            if(prevGreyImage.size() != currGreyImage.size()) {
                Timber.i("Warning, change of size prev: ${prevGreyImage.size()} cur: ${currGreyImage.size()}, dropping current OF")
                currGreyImage.copyTo(prevGreyImage)
                updatePoints(currGreyImage, scalingFactor)
                return arrayListOf()
            }
        }

        // If the number of flow points is too low, find new good features
        if (flowPtsCount < maxCorners / 2) {
            updatePoints(currGreyImage, scalingFactor)
        }

        // Run the KLT algorithm for Optical Flow
        try {
            Video.calcOpticalFlowPyrLK(prevGreyImage, currGreyImage, prevPts, currPts, status, err)
        } catch (e: Exception) {
            e.printStackTrace()
            Timber.e(e, "Error in optical flow")
            return arrayListOf()
        }
        flowPtsCount = 0
        val lines = createLines(scalingFactor)

        // update current image and points
        currGreyImage.copyTo(prevGreyImage)
        prevPts.fromArray(*currPts.toArray())

        return lines
    }

    private fun createLines(scalingFactor: Int): ArrayList<FloatArray> {
        val statusArr = status.toArray()
        val prevPtsArr = prevPts.toArray()
        val currPtsArr = currPts.toArray()
        val listLines = ArrayList<FloatArray>()
        for (i in 0 until prevPts.rows()) {
            if(i < statusArr.size && i < currPtsArr.size) {
                if (statusArr[i].toInt() == 1) {
                    val pt1x = prevPtsArr[i].x.toFloat() * scalingFactor
                    val pt1y = prevPtsArr[i].y.toFloat() * scalingFactor
                    val pt2x = currPtsArr[i].x.toFloat() * scalingFactor
                    val pt2y = currPtsArr[i].y.toFloat() * scalingFactor
                    listLines.add(
                        floatArrayOf(pt1x, pt1y, pt2x, pt2y)
                    )
                    flowPtsCount++
                }
            }
        }
        return listLines
    }

    fun moveDetectionsWithOF(mappedRecognitions: List<Detector.Recognition>, frameTimestamp:Long): MutableList<Detector.Recognition> {
        // Move detections with cumulative values of optical flows since frameTimestamps
        // returns a new detection list
        val movedRecognitions: MutableList<Detector.Recognition> = LinkedList()
        for (det in mappedRecognitions) {
            val rect = RectF(det.location)
            val center = PointF(rect.centerX(), rect.centerY())
            val move = PointF(0.0F, 0.0F)
            outputFlowLinesRollingArray.forEach { outputFlowLine: TrackingActivity.TimedOutputFlowLine ->
                if (outputFlowLine.timestamp >= frameTimestamp) {
                    val localMove = TrackerManager.calculateMedianFlowSpeedForTrack(
                        center,
                        outputFlowLine.data,
                        6
                    )
                    move.x += localMove?.x ?: 0.0F
                    move.y += localMove?.y ?: 0.0F
                }
            }
            val newLocation = RectF(
                det.location.left + move.x, det.location.top + move.y,
                det.location.right + move.x, det.location.bottom + move.y
            )
            val newDet = Detector.Recognition(
                det.classId,
                det.confidence,
                newLocation,
                det.maskIdx,
                det.mask,
                det.detectedClass,
                det.bitmap
            )
            movedRecognitions.add(newDet)

        }
        return movedRecognitions
    }
    fun inPlaceMoveDetectionsWithOF(mappedRecognitions: List<Detector.Recognition?>) {
        // Moves detections inplace with current optical flow value
        for (det in mappedRecognitions) {
            det?.let {
                val rect = RectF(it.location)
                val center = PointF(rect.centerX(), rect.centerY())
                val move =
                    TrackerManager.calculateMedianFlowSpeedForTrack(center, outputLinesFlow, 6)
                move?.let{ move ->
                    val newLocation = RectF(it.location.left + move.x, it.location.top + move.y,
                        it.location.right + move.x, it.location.bottom + move.y)
                    det.location = newLocation
                }
            }
        }
    }

    companion object {
        private const val FLOW_ROLLING_ARRAY_MAX_SIZE = 50
    }

}