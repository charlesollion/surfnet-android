package org.surfrider.surfnet.detection.tracking

import android.graphics.Point
import android.graphics.PointF
import android.graphics.RectF
import org.surfrider.surfnet.detection.tflite.Detector
import timber.log.Timber
import java.time.format.DateTimeFormatter
import java.util.*
import kotlin.math.abs

public class Tracker(det: TrackedDetection, idx: Int) {
    private val MAX_TIMESTAMP = 1000
    private val NUM_CONSECUTIVE_DET = 5
    private val ASSOCIATION_THRESHOLD = 10.0F

    var index = idx
    var status : TrackerStatus = TrackerStatus.RED
    var alreadyAssociated = false

    private val firstDetection = det
    private val trackedObjects: LinkedList<TrackedDetection> = LinkedList()
    var position = firstDetection.getCenter()

    init {
        trackedObjects.addLast(firstDetection)
    }
    
    private fun dist(p1: PointF, p2:PointF): Float {
        return kotlin.math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)).toFloat()
    }

    fun distTo(newPos: PointF): Float {
        return dist(position, newPos)
    }

    fun addDetection(newDet: TrackedDetection) {
        trackedObjects.addLast(newDet)
        position = newDet.getCenter()
        alreadyAssociated = true

        if(trackedObjects.size > NUM_CONSECUTIVE_DET) {
            status = TrackerStatus.GREEN
        }
    }

    fun update() {
        alreadyAssociated = false
        val age = compareTimeDifferenceInMilliseconds(System.currentTimeMillis(), trackedObjects.last.timestamp)
        Timber.i("AGE +> ${age.toString()} | MAX TIMESTAMP +> $MAX_TIMESTAMP")
        if(age > MAX_TIMESTAMP) {
            status = TrackerStatus.INACTIVE
        }
    }

    fun compareTimeDifferenceInMilliseconds(timestamp1: Long, timestamp2: Long): Long {
        return abs(timestamp1 - timestamp2)
    }

    class TrackedDetection(det: Detector.Recognition) {
        var location: RectF = RectF(det.location)
        var detectionConfidence = det.confidence
        var timestamp: Long = System.currentTimeMillis()
        var classId: String = det.id
        var associatedId = -1

        fun getCenter(): PointF {
            return PointF(location.centerX(), location.centerY())
        }
    }

    enum class TrackerStatus {
        GREEN, RED, INACTIVE
    }
}
