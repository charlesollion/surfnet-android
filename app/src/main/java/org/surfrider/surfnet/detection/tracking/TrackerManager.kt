package org.surfrider.surfnet.detection.tracking

import android.content.Context
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.VectorDrawable
import android.location.Location
import androidx.core.content.ContextCompat
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.surfrider.surfnet.detection.R
import org.surfrider.surfnet.detection.env.JsonFileWriter
import org.surfrider.surfnet.detection.env.MathUtils.calculateIoU
import org.surfrider.surfnet.detection.env.MathUtils.solveLinearSumAssignment
import org.surfrider.surfnet.detection.models.TrackerPosition
import org.surfrider.surfnet.detection.models.TrackerResult
import org.surfrider.surfnet.detection.models.TrackerTrash
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.min

class TrackerManager {

    val trackers: LinkedList<Tracker> = LinkedList<Tracker>()
    val detectedWaste: LinkedList<Tracker> = LinkedList<Tracker>()
    val positions: ArrayList<TrackerPosition> = ArrayList()

    var displayDetection = true

    private var bmpYellow: Bitmap? = null
    private var bmpGreen: Bitmap? = null
    private var bmpWhite: Bitmap? = null

    private var trackerIndex = 0

    fun updateTrackers() {
        trackers.forEach { tracker -> tracker.update() }
    }

    fun addPosition(location: Location?, date: String) {
        location?.let {
            val position =
                TrackerPosition(lat = it.latitude, lng = it.longitude, date = date)

            positions.add(position)
        }
    }

    @Synchronized
    fun processDetections(results: List<Recognition>, location: Location?) {
        if (results.isEmpty()) {
            return
        }
        // Store all Recognition objects in a list of TrackedDetections
        val dets = LinkedList<Tracker.TrackedDetection>()
        for (result in results) {
            dets.addLast(Tracker.TrackedDetection(result))
        }

        // Create cost matrix
        if (trackers.size > 0) {
            val costMatrix = Array(dets.size) { _ ->
                DoubleArray(trackers.size)
            }
            dets.forEachIndexed { t, det ->
                trackers.forEachIndexed { i, tracker ->
                    costMatrix[t][i] = cost(det, tracker)
                }
            }

            // Compute best assignment
            val assignments = solveLinearSumAssignment(costMatrix)
            for ((detIdx, trackIdx) in assignments) {
                val detection = dets[detIdx]
                val tracker = trackers[trackIdx]
                tracker.addDetection(detection)
                detection.associatedId = trackIdx
            }
        }
        // create new trackers for unassigned detections
        for (det in dets) {
            if (det.associatedId == -1) {
                trackers.addLast(Tracker(det, trackerIndex, location))
                trackerIndex++
            }
        }
    }


    private fun cost(det: Tracker.TrackedDetection, tracker: Tracker): Double {
        if (tracker.status != Tracker.TrackerStatus.INACTIVE && !tracker.alreadyAssociated) {
            val dist = tracker.distTo(det.getCenter()).toDouble() / SCREEN_DIAGONAL
            if (dist > ASSOCIATION_THRESHOLD) {
                return Double.MAX_VALUE
            }
            val confidence = 1.0 - det.detectionConfidence
            val iou = 1.0 - calculateIoU(det.rect, tracker.trackedObjects.last.rect)
            val classMatch = if (det.classId == tracker.trackedObjects.last.classId) 0.0 else 1.0
            val strength = 1.0 - tracker.strength
            val cost =
                W_DIST * dist + W_CONF * confidence + W_IOU * iou + W_CLASS * classMatch + W_TRACKER_STRENGTH * strength
            // Timber.i("${tracker.index}/${det.rect}:$cost --- dist:${dist} confidence:${confidence} iou:$iou classmatch:$classMatch strength:${strength}")
            return cost

        }
        return Double.MAX_VALUE
    }

    @Synchronized
    fun draw(
            canvas: Canvas,
            context: Context?,
            previewWidth: Int,
            previewHeight: Int,
            showOF: Boolean
    ) {
        // Build transform matrix from canvas and context
        val frameToCanvasTransform = Matrix()
        val scale = min(
                canvas.width / previewWidth.toFloat(), canvas.height / previewHeight.toFloat()
        )
        frameToCanvasTransform.postScale(scale, scale)
        if (bmpYellow == null) {
            bmpYellow = context?.let { getBitmap(it, R.drawable.yellow_dot) }
        }
        if (bmpGreen == null) {
            bmpGreen = context?.let { getBitmap(it, R.drawable.check_icon) }
        }
        if (bmpWhite == null) {
            bmpWhite = context?.let { getBitmap(it, R.drawable.yellow_dashed_circle) }
        }

        for (tracker in trackers) {
            val trackedPos = tracker.position
            //Only draw tracker if not inactive
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                var bmp: Bitmap? = null
                if (displayDetection || !(tracker.status == Tracker.TrackerStatus.RED || tracker.status == Tracker.TrackerStatus.LOADING)) {
                    bmp = if (tracker.status == Tracker.TrackerStatus.GREEN) {
                        if (!detectedWaste.contains(tracker)) {
                            detectedWaste.add(tracker)
                        }
                        bmpGreen
                    } else {
                        if (tracker.status == Tracker.TrackerStatus.LOADING)
                            bmpYellow
                        else
                            bmpWhite
                    }

                }

                // Draw the speed line to show displacement of the tracker depending on camera motion
                if (showOF) {
                    drawOF(canvas, tracker, frameToCanvasTransform)
                }

                if (bmp != null) {
                    val bmpWidth = bmp.width.div(scale)
                    val bmpHeight = bmp.height.div(scale)

                    val point =
                            floatArrayOf(trackedPos.x - bmpWidth / 2, trackedPos.y - bmpHeight / 2)
                    frameToCanvasTransform.mapPoints(point)

                    canvas.drawBitmap(bmp, point[0], point[1], null)

                    // Draw text with tracker number
                    val paint = Paint()
                    paint.textSize = 40.0F
                    canvas.drawText(tracker.index.toString(), point[0], point[1], paint)

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
                }
            }
        }
    }

    private fun drawOF(canvas: Canvas, tracker: Tracker, transform: Matrix) {
        val speedLine = floatArrayOf(
                tracker.position.x,
                tracker.position.y,
                tracker.position.x + tracker.speed.x,
                tracker.position.y + tracker.speed.y
        )
        transform.mapPoints(speedLine)
        val paintLine = Paint()
        paintLine.color = Color.GREEN
        paintLine.strokeWidth = 8.0F
        canvas.drawLines(speedLine, paintLine)
    }

    private fun drawEllipses(canvas: Canvas, tracker: Tracker, transform: Matrix) {
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4.0F
        paint.color = Color.BLUE
        val dims = PointF((1.0F + tracker.speedCov.x * 0.0F) * ASSOCIATION_THRESHOLD * SCREEN_DIAGONAL,
                (1.0F + tracker.speedCov.y * 0.0F) * ASSOCIATION_THRESHOLD * SCREEN_DIAGONAL)
        val rect = RectF(tracker.position.x - dims.x, tracker.position.y - dims.y,
                tracker.position.x + dims.x, tracker.position.y + dims.y)
        transform.mapRect(rect)
        canvas.drawOval(rect, paint)
    }

    fun getCurrentRois(width: Int, height: Int, downScale: Int, squareSize: Int): Mat? {
        // Get regions of interest within the frame: areas around each tracker
        // The output is a mask matrix with 1s next to tracker centers and 0s otherwise
        if (trackers.size == 0) {
            return null
        }
        val currRois = Mat.zeros(height / downScale, width / downScale, CvType.CV_8UC1)
        for (tracker in trackers) {
            val trackedPos = tracker.position
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                val xCenter: Int = trackedPos.x.toInt() / downScale
                val yCenter: Int = trackedPos.y.toInt() / downScale

                for (i in -squareSize / 2..squareSize / 2) {
                    for (j in -squareSize / 2..squareSize / 2) {
                        val x = xCenter + i
                        val y = yCenter + j

                        if (x in 0..<width && y >= 0 && y < height) {
                            currRois.put(y, x, byteArrayOf(1))
                        }
                    }
                }
            }
        }

        // Add a central region always available for tracking
        val centerH = height / downScale / 2
        val centerW = width / downScale / 2
        val centerSquareSize = 40
        for (i in -centerSquareSize / 2..centerSquareSize / 2) {
            for (j in -centerSquareSize / 2..centerSquareSize / 2) {
                currRois.put(centerH + i, centerW + j, byteArrayOf(1))
            }
        }

        return currRois
    }

    fun associateFlowWithTrackers(listOfFlowLines: ArrayList<FloatArray>, flowRefreshRateInMillis: Long): PointF {
        // Associate each tracker with flow speed

        // Compute the average flow for debug purposes
        val avgMotionSpeed = PointF(0.0F, 0.0F)
        if (listOfFlowLines.size > 0) {
            for (line in listOfFlowLines) {
                avgMotionSpeed.x += (line[2] - line[0])
                avgMotionSpeed.y += (line[3] - line[1])
            }
            avgMotionSpeed.x /= listOfFlowLines.size * flowRefreshRateInMillis / 1000.0F
            avgMotionSpeed.y /= listOfFlowLines.size * flowRefreshRateInMillis / 1000.0F
        }

        for (tracker in trackers) {
            val medianSpeed = calculateMedianFlowSpeedForTrack(tracker.position, listOfFlowLines, 6)

            // scale speed depending on optical flow refresh rate
            /*medianSpeed?.let {
                it.x /= flowRefreshRateInMillis / 1000.0F
                it.y /= flowRefreshRateInMillis / 1000.0F
            }*/

            tracker.updateSpeed(medianSpeed
                    ?: avgMotionSpeed, ASSOCIATION_THRESHOLD * SCREEN_DIAGONAL)

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
                    it.intrinsicWidth, it.intrinsicHeight, Bitmap.Config.ARGB_8888
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

    fun sendData(context: Context, email: String?) {
        val trashes = ArrayList<TrackerTrash>()
        trackers.forEach {
            if (it.isValid()) {
                val date = Date(it.startDate)
                val iso8601Format =
                        SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.getDefault())
                val iso8601DateString = iso8601Format.format(date)
                trashes.add(
                        TrackerTrash(
                                date = iso8601DateString,
                                lat = it.location?.latitude,
                                lng = it.location?.longitude,
                                name = it.computeMajorityClass()
                        )
                )
            }

            val result = TrackerResult(
                    move = null,
                    bank = null,
                    trackingMode = "automatic",
                    files = ArrayList(),
                    trashes = trashes,
                    positions = positions,
                    comment = "email : $email"
            )
            val actualDate = Calendar.getInstance().time
            val saveFileDateFormat = SimpleDateFormat("yyyyMMddHHmmss", Locale.getDefault())
            val saveFileDateString = saveFileDateFormat.format(actualDate)
            //Save JSON file to "Downloads" folder
            JsonFileWriter.writeResultToJsonFile(context, result, saveFileDateString)
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
                    .map { PointF(it[2] - it[0], it[3] - it[1]) }
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
        private fun dist(p1: FloatArray, p2: PointF): Float {
            return kotlin.math.sqrt((p1[0] - p2.x) * (p1[0] - p2.x) + (p1[1] - p2.y) * (p1[1] - p2.y))
        }

        private const val SCREEN_DIAGONAL = 960.0F // sqrt(720x1280)
        private const val ASSOCIATION_THRESHOLD = 40.0F / SCREEN_DIAGONAL

        // Weights of different scores
        private const val W_DIST = 1.0
        private const val W_CONF = 0.1
        private const val W_IOU = 1.0
        private const val W_CLASS = 0.1
        private const val W_TRACKER_STRENGTH = 0.1
    }
}