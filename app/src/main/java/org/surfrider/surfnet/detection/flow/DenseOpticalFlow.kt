package org.surfrider.surfnet.detection.flow

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import timber.log.Timber


class DenseOpticalFlow {
    private var currGreyImage = Mat()
    private var prevGreyImage = Mat()
    private var prevPts = MatOfPoint2f()
    private var currPts = MatOfPoint2f()
    private val maxCorners = 50
    private var flowPtsCount = 50
    private var status = MatOfByte()
    private var err = MatOfFloat()

    private fun updatePoints(greyImage: Mat) {
        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(greyImage, corners, maxCorners, 0.1, 5.0)
        prevPts.fromArray(*corners.toArray())
    }

    fun run(newFrame: Mat): Point {
        Timber.i("Optical Flow - Start")
        // convert the frame to Gray
        Imgproc.cvtColor(newFrame, currGreyImage, Imgproc.COLOR_RGBA2GRAY)
        // if this is the first loop, find good features
        if (prevGreyImage.empty()) {
            currGreyImage.copyTo(prevGreyImage)
            updatePoints(currGreyImage)
            return Point(0.0, 0.0)
        }
        // If the number of flow points is too low, find new good features
        if (flowPtsCount < maxCorners / 5) {
            currGreyImage.copyTo(prevGreyImage)
            updatePoints(currGreyImage)
        }
        // Run the KLT algorithm for Optical Flow
        Video.calcOpticalFlowPyrLK(prevGreyImage, currGreyImage, prevPts, currPts, status, err)

        // returns the average motion vector
        val avgMV = avgMotionVector()
        Timber.i("Optical Flow - Done algorithm, tracked points: "+flowPtsCount)
        return avgMV
    }

    private fun avgMotionVector() : Point {
        val statusArr = status.toArray()
        val prevPtsArr = prevPts.toArray()
        val currPtsArr = currPts.toArray()
        var pt1Avg = Point(0.0, 0.0)
        var pt2Avg = Point(0.0, 0.0)
        for (i in 0 until prevPts.rows()) {
            if (statusArr[i].toInt() == 1) {
                val pt1x = prevPtsArr[i].x
                val pt1y = prevPtsArr[i].y
                val pt2x = currPtsArr[i].x
                val pt2y = currPtsArr[i].y
                pt1Avg.x += pt1x
                pt1Avg.y += pt1y
                pt2Avg.x += pt2x
                pt2Avg.y += pt2y
                flowPtsCount++
            }
        }

        // Calculate the average motion vector
        pt1Avg.x *= 1.0 / flowPtsCount.toDouble()
        pt1Avg.y *= 1.0 / flowPtsCount.toDouble()
        pt2Avg.x *= 1.0 / flowPtsCount.toDouble()
        pt2Avg.y *= 1.0 / flowPtsCount.toDouble()
        return Point(pt2Avg.x - pt1Avg.x, pt2Avg.y - pt1Avg.y)
    }


}