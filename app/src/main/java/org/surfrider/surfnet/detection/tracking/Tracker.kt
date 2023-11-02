package org.surfrider.surfnet.detection.tracking

import android.graphics.PointF
import android.graphics.RectF
import android.location.Location
import org.surfrider.surfnet.detection.tflite.Detector
import java.util.*
import kotlin.math.abs

class Tracker(det: TrackedDetection, idx: Int, lctn: Location?) {

    var index = idx
    var location: Location? = lctn
    var status : TrackerStatus = TrackerStatus.RED
    var animation = false
    var alreadyAssociated = false

    private var lastUpdatedTimestamp: Long = 0
    private var animationTimeStamp : Long? = null

    private val firstDetection = det
    val trackedObjects: LinkedList<TrackedDetection> = LinkedList()
    var position = firstDetection.getCenter()
    var speed = PointF(0.0F, 0.0F)
    var strength = 0.0F

    init {
        trackedObjects.addLast(firstDetection)
    }

    fun distTo(newPos: PointF): Float {
        return dist(position, newPos)
    }

    fun addDetection(newDet: TrackedDetection) {
        trackedObjects.addLast(newDet)
        position = newDet.getCenter()
        strength += 0.2F
        if(strength > 1.0F) {
            strength = 1.0F
        }
        alreadyAssociated = true

        if(trackedObjects.size > NUM_CONSECUTIVE_DET && status == TrackerStatus.RED) {
            status = TrackerStatus.GREEN
            animation = true
            animationTimeStamp = newDet.timestamp
        }
    }

    fun updateSpeed(measuredSpeed: PointF) {
        // Move tracker directly
        position.x += measuredSpeed.x
        position.y += measuredSpeed.y
        speed = measuredSpeed
    }


    fun update() {
        alreadyAssociated = false

        val currTimeStamp = System.currentTimeMillis()
        val age = timeDiffInMilli(currTimeStamp, trackedObjects.last.timestamp)
        // Timber.i("AGE +> ${age.toString()} | MAX TIMESTAMP +> $MAX_TIMESTAMP")

        val ageOfAnimation = animationTimeStamp?.let {
            timeDiffInMilli(currTimeStamp,
                it
            )
        }
        if (ageOfAnimation != null) {
            if (ageOfAnimation > MAX_ANIMATION_TIMESTAMP) {
                animation = false
            }
        }

        // Green last twice as long as red
        if((status == TrackerStatus.RED && age > MAX_TIMESTAMP) || (status == TrackerStatus.GREEN && age > MAX_TIMESTAMP * 2)) {
            status = TrackerStatus.INACTIVE
        }

        lastUpdatedTimestamp = currTimeStamp
    }

    private fun timeDiffInMilli(timestamp1: Long, timestamp2: Long): Long {
        return abs(timestamp1 - timestamp2)
    }

    class TrackedDetection(det: Detector.Recognition) {
        var rect: RectF = RectF(det.location)
        var detectionConfidence: Float = det.confidence
        var timestamp: Long = System.currentTimeMillis()
        var classId: String = det.id
        var associatedId = -1

        fun getCenter(): PointF {
            return PointF(rect.centerX(), rect.centerY())
        }
    }

    enum class TrackerStatus {
        GREEN, RED, INACTIVE
    }

    companion object {
        @JvmStatic
        private fun dist(p1: PointF, p2:PointF): Float {
            return kotlin.math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))
        }

        private const val MAX_TIMESTAMP = 2000
        private const val MAX_ANIMATION_TIMESTAMP = 1000
        private const val NUM_CONSECUTIVE_DET = 5
    }
}
