package org.surfrider.surfnet.detection.flow

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video


class DenseOpticalFlow {
    private var currGreyImage = Mat()
    private var prevGreyImage = Mat()
    private var prevPts = MatOfPoint2f()
    private var currPts = MatOfPoint2f()
    private val maxCorners = 50
    private val minDistance = 10.0
    private var flowPtsCount = 50
    private var status = MatOfByte()
    private var err = MatOfFloat()


    private fun updatePoints(greyImage: Mat, mask: Mat?) {
        val corners = MatOfPoint()
        var numberCorners = maxCorners
        if(!prevPts.empty())
            numberCorners -= prevPts.rows()
        Imgproc.goodFeaturesToTrack(greyImage, corners, numberCorners, 0.1, minDistance, mask ?: Mat())

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


    fun run(newFrame: Mat, mask:Mat?, scalingFactor: Int): ArrayList<FloatArray> {
        return  pyrLK(newFrame, mask, scalingFactor)
        // return farneback(newFrame)
    }

    private fun pyrLK(newFrame: Mat, mask: Mat?, scalingFactor: Int) : ArrayList<FloatArray> {
        // convert the frame to Gray
        Imgproc.cvtColor(newFrame, currGreyImage, Imgproc.COLOR_RGBA2GRAY)
        // if this is the first loop, find good features
        /*if(mask == null || mask.empty()) {
            Timber.i("OF mask null or empty")
        }*/
        // Timber.i("imgSize: ${currGreyImage.size().toString()} mask size: ${mask?.size().toString()}")
        if (prevGreyImage.empty()) {
            currGreyImage.copyTo(prevGreyImage)
            updatePoints(currGreyImage, mask)
            return arrayListOf()
        }
        // If the number of flow points is too low, find new good features
        if (flowPtsCount < maxCorners / 2) {
            // currGreyImage.copyTo(prevGreyImage)
            updatePoints(currGreyImage, mask)
        }
        // Run the KLT algorithm for Optical Flow
        Video.calcOpticalFlowPyrLK(prevGreyImage, currGreyImage, prevPts, currPts, status, err)
        //Timber.i("output of flow -- prevPts: ${prevPts.size()} currPts: ${currPts.size()} status: ${status.size()}")
        flowPtsCount = 0
        val lines = createLines(scalingFactor)

        // update last image
        currGreyImage.copyTo(prevGreyImage)
        prevPts.fromArray(*currPts.toArray())

        // returns the average motion vector
        // val avgMV = avgMotionVector()

        // returns the lines
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
        // Timber.i("output line flow size in func: ${listLines.size}")
        return listLines
    }


}