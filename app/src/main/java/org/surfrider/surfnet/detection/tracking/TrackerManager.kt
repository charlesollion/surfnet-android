package org.surfrider.surfnet.detection.tracking

import android.content.Context
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.VectorDrawable
import androidx.core.content.ContextCompat
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.surfrider.surfnet.detection.R
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import timber.log.Timber
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.min

class TrackerManager {
    private val ASSOCIATION_THRESHOLD = 50.0F

    val trackers: LinkedList<Tracker> = LinkedList<Tracker>()
    private var trackerIndex = 0
    val detectedWaste: LinkedList<Tracker> = LinkedList<Tracker>()

    fun updateTrackers() {
        trackers.forEach { tracker -> tracker.update() }
    }

    @Synchronized
    fun processDetections(results: List<Recognition>) {
        // Store all Recognition objects in a list of TrackedDetections
        val dets = LinkedList<Tracker.TrackedDetection>()
        for (result in results) {
            dets.addLast(Tracker.TrackedDetection(result))
        }

        // Associate results with current trackers
        for (det in dets) {
            val position = det.getCenter()
            var minDist = 10000.0F
            // Greedy assignment of trackers
            trackers.forEachIndexed { i, tracker ->
                if (tracker.status != Tracker.TrackerStatus.INACTIVE && !tracker.alreadyAssociated) {
                    val dist = tracker.distTo(position)
                    if (dist < minDist && dist < ASSOCIATION_THRESHOLD) {
                        minDist = dist
                        det.associatedId = i
                    }
                }

            }
            if (det.associatedId != -1) {
                trackers[det.associatedId].addDetection(det)
            } else {
                trackers.addLast(Tracker(det, trackerIndex, 0.1))
                trackerIndex++
            }
        }
    }

    @Synchronized
    fun draw(canvas: Canvas, context: Context?, previewWidth: Int, previewHeight: Int, showOF: Boolean) {
        // Build transform matrix from canvas and context
        val frameToCanvasTransform = Matrix()
        val scale = min(
            canvas.width / previewWidth.toFloat(), canvas.height / previewHeight.toFloat()
        )
        frameToCanvasTransform.postScale(scale, scale)

        for (tracker in trackers) {
            val trackedPos = tracker.position
            //Only draw tracker if not inactive
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                val bmp = context?.let {
                    getBitmap(
                        it,
                        if (tracker.status == Tracker.TrackerStatus.GREEN) {
                            if (!detectedWaste.contains(tracker)) {
                                detectedWaste.add(tracker)
                            }
                            R.drawable.green_dot
                        } else {
                            R.drawable.red_dot
                        }
                    )
                } ?: return

                val bmpWidth = bmp.width.div(scale)
                val bmpHeight = bmp.height.div(scale)

                val point =
                    floatArrayOf(trackedPos.x - bmpWidth / 2, trackedPos.y - bmpHeight / 2)
                frameToCanvasTransform.mapPoints(point)

                canvas.drawBitmap(bmp, point[0], point[1], null)


                // Draw the speed line to show displacement of the tracker depending on camera motion
                if(showOF) {
                    val speedLine = floatArrayOf(
                        trackedPos.x,
                        trackedPos.y,
                        trackedPos.x + tracker.speed.x,
                        trackedPos.y + tracker.speed.y
                    )
                    frameToCanvasTransform.mapPoints(speedLine)
                    val paintLine = Paint()
                    paintLine.color = Color.GREEN
                    paintLine.strokeWidth = 8.0F
                    canvas.drawLines(speedLine, paintLine)
                }

                //Animation drawing
                if (tracker.animation) {
                    val shouldShowBottomAnimation = trackedPos.y < canvas.height.div(scale) / 2
                    val animation = context?.let {
                        getBitmap(
                            it,
                            if (shouldShowBottomAnimation) R.drawable.animation_down else R.drawable.animation
                        )
                    } ?: return

                    val animationWidth = animation.width.div(scale)
                    val animationHeight = animation.height.div(scale)

                    val animationPoint = floatArrayOf(
                        trackedPos.x - (animationWidth / 2) + 3,
                        if (shouldShowBottomAnimation) trackedPos.y + bmpHeight / 2 else trackedPos.y - bmpHeight / 2 - (animationHeight)
                    )
                    frameToCanvasTransform.mapPoints(animationPoint)
                    canvas.drawBitmap(
                        animation,
                        animationPoint[0],
                        animationPoint[1],
                        null
                    )
                }

                //affichage du text avec le numéro du tracker
                val paint = Paint()
                paint.textSize = 40.0F
                canvas.drawText(tracker.index.toString(), point[0], point[1], paint)

            }
        }
    }

    fun getCurrentRois(width: Int, height: Int, downScale: Int, squareSize: Int): Mat? {
        // Get regions of interest within the frame: areas around each tracker
        // The output is a mask matrix with 1s next to tracker centers and 0s otherwise
        if(trackers.size == 0) {
            return null
        }
        val currRois = Mat.zeros(height / downScale, width / downScale, CvType.CV_8UC1)
        for (tracker in trackers) {
            val trackedPos = tracker.position
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                val xCenter: Int = trackedPos.x.toInt() / downScale
                val yCenter: Int = trackedPos.y.toInt() / downScale

                for (i in -squareSize/2..squareSize/2) {
                    for (j in -squareSize/2..squareSize/2) {
                        val x = xCenter + i
                        val y = yCenter + j

                        if (x >= 0 && x < width && y >= 0 && y < height) {
                            currRois.put(y, x, byteArrayOf(1))
                        }
                    }
                }
            }
        }
        return currRois
    }

    fun associateFlowWithTrackers(listOfFlowLines: ArrayList<FloatArray>, flowRefreshRateInMillis: Long): PointF {
        // Associate each tracker with flow speed
        // V1: just take the average flow
        var avgMotionSpeed = PointF(0.0F, 0.0F)
        if(listOfFlowLines.size > 0) {
            for (line in listOfFlowLines) {
                avgMotionSpeed.x += (line[2] - line[0])
                avgMotionSpeed.y += (line[3] - line[1])
            }
            avgMotionSpeed.x /= listOfFlowLines.size * flowRefreshRateInMillis / 1000.0F
            avgMotionSpeed.y /= listOfFlowLines.size * flowRefreshRateInMillis / 1000.0F
        }

        // Timber.i("Motion Speed: ${motionSpeed.x} ${motionSpeed.y}")
        for(tracker in trackers) {
            var medianSpeed = calculateMedianFlowSpeedForTrack(tracker.position, listOfFlowLines, 6)
            medianSpeed?.let {
                it.x /= 1.0F //flowRefreshRateInMillis / 1000.0F
                it.y /= 1.0F //flowRefreshRateInMillis / 1000.0F
            }

            Timber.i("${tracker.index}: Motion Speed: ${medianSpeed?.x} ${medianSpeed?.y}")
            tracker.updateSpeed(medianSpeed?:avgMotionSpeed)
        }
        return avgMotionSpeed
    }

    @Synchronized
    fun drawDebug(canvas: Canvas) {
        for (tracker in trackers) {
            val trackedPos = tracker.position
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                val paint = Paint()
                paint.textSize = 40.0F
                canvas.drawText(tracker.index.toString(), trackedPos.x, trackedPos.y, paint)
            }
        }
    }

    private fun getBitmap(vectorDrawable: VectorDrawable?): Bitmap? {
        vectorDrawable?.let {
            val bitmap = Bitmap.createBitmap(
                it.intrinsicWidth * 2, it.intrinsicHeight * 2, Bitmap.Config.ARGB_8888
            )
            val canvas = Canvas(bitmap)
            it.setBounds(0, 0, canvas.width, canvas.height)
            it.draw(canvas)
            return bitmap
        }
        return null
    }

    private fun getBitmap(context: Context, drawableId: Int): Bitmap? {
        return when (val drawable = ContextCompat.getDrawable(context, drawableId)) {
            is BitmapDrawable -> {
                BitmapFactory.decodeResource(context.resources, drawableId)
            }

            is VectorDrawable -> {
                getBitmap(drawable as VectorDrawable?)
            }

            else -> {
                throw IllegalArgumentException("unsupported drawable type")
            }
        }
    }

    companion object {
        @JvmStatic
        fun calculateMedianFlowSpeedForTrack(
            trackPoint: PointF,
            opticalFlowData: ArrayList<FloatArray>,
            k: Int
        ): PointF? {
            val kNearestOpticalFlowPoints = findKNearestOpticalFlowPoints(trackPoint, opticalFlowData, k)

            if (kNearestOpticalFlowPoints.isNotEmpty()) {
                // Calculate the median flow speed of the k nearest points
                return calculateMedianFlowSpeed(kNearestOpticalFlowPoints)
            }

            return null // No data points found
        }

        @JvmStatic
        private fun findKNearestOpticalFlowPoints(
            trackPoint: PointF,
            opticalFlowData: ArrayList<FloatArray>,
            k: Int
        ): List<PointF> {
            return opticalFlowData
                .sortedBy { dist(it, trackPoint) }
                .map { PointF(it[2]-it[0], it[3]-it[1]) }
                .take(k)
        }

        @JvmStatic
        private fun calculateMedianFlowSpeed(opticalFlowPoints: List<PointF>): PointF {
            val sortedFlowSpeedX = opticalFlowPoints.map { it.x }.sorted()
            val sortedFlowSpeedY = opticalFlowPoints.map { it.y }.sorted()

            val medianFlowSpeedX = if (sortedFlowSpeedX.size % 2 == 1) {
                sortedFlowSpeedX[sortedFlowSpeedX.size / 2]
            } else {
                (sortedFlowSpeedX[sortedFlowSpeedX.size / 2 - 1] + sortedFlowSpeedX[sortedFlowSpeedX.size / 2]) / 2.0
            }

            val medianFlowSpeedY = if (sortedFlowSpeedY.size % 2 == 1) {
                sortedFlowSpeedY[sortedFlowSpeedY.size / 2]
            } else {
                (sortedFlowSpeedY[sortedFlowSpeedY.size / 2 - 1] + sortedFlowSpeedY[sortedFlowSpeedY.size / 2]) / 2.0
            }

            return PointF(medianFlowSpeedX.toFloat(), medianFlowSpeedY.toFloat())
        }
        @JvmStatic
        private fun dist(p1: FloatArray, p2:PointF): Float {
            return kotlin.math.sqrt((p1[0] - p2.x) * (p1[0] - p2.x) + (p1[1] - p2.y) * (p1[1] - p2.y))
        }
        @JvmStatic
        private fun dist(p1: PointF, p2:PointF): Float {
            return kotlin.math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))
        }
    }

}