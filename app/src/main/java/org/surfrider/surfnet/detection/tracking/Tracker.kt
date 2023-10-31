package org.surfrider.surfnet.detection.tracking

import android.graphics.PointF
import android.graphics.RectF
import android.location.Location
import org.surfrider.surfnet.detection.tflite.Detector
import java.util.*
import kotlin.math.abs

public class Tracker(det: TrackedDetection, idx: Int, lctn: Location?) {
    private val MAX_TIMESTAMP = 3000
    private val MAX_ANIMATION_TIMESTAMP = 1000
    private val NUM_CONSECUTIVE_DET = 5

    var index = idx
    var location: Location? = lctn
    var status : TrackerStatus = TrackerStatus.RED
    var animation = false
    var alreadyAssociated = false
    var lastUpdatedTimestamp: Long = 0

    private var animationTimeStamp : Long? = null

    private val firstDetection = det
    private val trackedObjects: LinkedList<TrackedDetection> = LinkedList()
    var position = firstDetection.getCenter()
    var speed = PointF(0.0F, 0.0F)

    init {
        trackedObjects.addLast(firstDetection)
    }
    
    private fun dist(p1: PointF, p2:PointF): Float {
        return kotlin.math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))
    }

    fun distTo(newPos: PointF): Float {
        return dist(position, newPos)
    }

    fun addDetection(newDet: TrackedDetection) {
        trackedObjects.addLast(newDet)
        position = newDet.getCenter()
        alreadyAssociated = true

        if(trackedObjects.size > NUM_CONSECUTIVE_DET && status == TrackerStatus.RED) {
            status = TrackerStatus.GREEN
            animation = true
            animationTimeStamp = newDet.timestamp
        }
    }

    fun update(flowRefreshRateInMillis: Long) {
        alreadyAssociated = false

        val currTimeStamp = System.currentTimeMillis()
        val age = compareTimeDifferenceInMilliseconds(currTimeStamp, trackedObjects.last.timestamp)
        // Timber.i("AGE +> ${age.toString()} | MAX TIMESTAMP +> $MAX_TIMESTAMP")

        val ageOfAnimation = animationTimeStamp?.let {
            compareTimeDifferenceInMilliseconds(currTimeStamp,
                it
            )
        }
        if (ageOfAnimation != null) {
            if (ageOfAnimation > MAX_ANIMATION_TIMESTAMP) {
                animation = false
            }
        }

        if(age > MAX_TIMESTAMP) {
            status = TrackerStatus.INACTIVE
        }

        // Move tracker
        if(lastUpdatedTimestamp > 0) {
            val elapsedTime = compareTimeDifferenceInMilliseconds(currTimeStamp, lastUpdatedTimestamp)
            position.x += speed.x * (elapsedTime / flowRefreshRateInMillis)
            position.y += speed.y * (elapsedTime / flowRefreshRateInMillis)
        }
        lastUpdatedTimestamp = currTimeStamp
    }

    private fun compareTimeDifferenceInMilliseconds(timestamp1: Long, timestamp2: Long): Long {
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
