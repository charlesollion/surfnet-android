package org.surfrider.surfnet.detection.flow

import android.graphics.Canvas
import org.opencv.core.*
import org.opencv.core.Core.multiply
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

    private fun updatePoints(greyImage: Mat, mask: Mat?) {
        Timber.i("updating OF points")
        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(greyImage, corners, maxCorners, 0.1, 5.0, mask?:Mat())
        prevPts.fromArray(*corners.toArray())
    }

    fun run(newFrame: Mat, mask:Mat?): ArrayList<FloatArray> {
        return  PyrLK(newFrame, mask)
        // return Farneback(newFrame)
    }

    private fun PyrLK(newFrame: Mat, mask: Mat?) : ArrayList<FloatArray> {
        // convert the frame to Gray
        Imgproc.cvtColor(newFrame, currGreyImage, Imgproc.COLOR_RGBA2GRAY)
        // if this is the first loop, find good features
        // Timber.i("imgSize: ${currGreyImage.size().toString()} mask size: ${mask?.size().toString()}")
        if (prevGreyImage.empty()) {
            currGreyImage.copyTo(prevGreyImage)
            updatePoints(currGreyImage, mask)
            return arrayListOf()
            // return Point(0.0, 0.0)
        }
        // If the number of flow points is too low, find new good features
        if (flowPtsCount < maxCorners / 2) {
            // currGreyImage.copyTo(prevGreyImage)
            updatePoints(currGreyImage, mask)
        }
        // Run the KLT algorithm for Optical Flow
        Video.calcOpticalFlowPyrLK(prevGreyImage, currGreyImage, prevPts, currPts, status, err)
        Timber.i("output of flow -- prevPts: ${prevPts.size()} currPts: ${currPts.size()} status: ${status.size()}")
        flowPtsCount = 0
        val lines = createLines()

        // update last image
        currGreyImage.copyTo(prevGreyImage)
        prevPts.fromArray(*currPts.toArray())

        // returns the average motion vector
        // val avgMV = avgMotionVector()

        // returns the lines
        return lines
    }

    private fun Farneback(newFrame: Mat) : Mat {
        // Downsample and convert to Gray
        val downscaledFrame = Mat()
        Imgproc.resize(newFrame, downscaledFrame, Size(), 0.5, 0.5)
        Imgproc.cvtColor(downscaledFrame, currGreyImage, Imgproc.COLOR_RGBA2GRAY)

        val flow = Mat(currGreyImage.size(), CvType.CV_32FC2)

        if (prevGreyImage.empty()) {
            currGreyImage.copyTo(prevGreyImage)
            return flow
            //return Point(0.0, 0.0)
        }
        Video.calcOpticalFlowFarneback(prevGreyImage, currGreyImage, flow,
            0.5, 3, 15, 3, 5, 1.2, Video.OPTFLOW_FARNEBACK_GAUSSIAN)

        val kernel = Mat.ones(3,3, CvType.CV_32FC1)
        val factor = Scalar(0.11111111)
        Core.multiply(kernel, factor, kernel)
        val flow2 = Mat(flow.size(), CvType.CV_32FC2)
        Imgproc.filter2D(flow, flow2, flow.depth(), kernel)
        Imgproc.resize(flow2, flow2, Size(), 0.25, 0.25, Imgproc.INTER_LINEAR)
        if(flow2 != null)
            Core.transpose(flow2, flow2)
        //val avgMV : Scalar = Core.mean(flow)
        //Point(avgMV.`val`[0], avgMV.`val`[1])

        return flow2
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

    private fun createLines(): ArrayList<FloatArray> {
        val statusArr = status.toArray()
        val prevPtsArr = prevPts.toArray()
        val currPtsArr = currPts.toArray()
        val listLines = ArrayList<FloatArray>()
        for (i in 0 until prevPts.rows()) {
            if(i < statusArr.size && i < currPtsArr.size) {
                if (statusArr[i].toInt() == 1) {
                    val pt1x = prevPtsArr[i].x.toFloat()
                    val pt1y = prevPtsArr[i].y.toFloat()
                    val pt2x = currPtsArr[i].x.toFloat()
                    val pt2y = currPtsArr[i].y.toFloat()
                    listLines.add(
                        floatArrayOf(pt1x, pt1y, pt2x, pt2y)
                    )
                    flowPtsCount++
                }
            }
        }
        // Timber.i("output line flow size in func: ${listLines.size}")
        return listLines
    }


}