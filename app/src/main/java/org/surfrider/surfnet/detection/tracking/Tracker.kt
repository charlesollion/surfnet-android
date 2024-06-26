package org.surfrider.surfnet.detection.tracking

import android.graphics.PointF
import android.graphics.RectF
import android.location.Location
import org.surfrider.surfnet.detection.env.MathUtils.malDist
import org.surfrider.surfnet.detection.tflite.Detector
import java.util.*
import kotlin.math.abs
import kotlin.math.min

class Tracker(det: TrackedDetection, idx: Int, lctn: Location?) {
    var index = idx
    var location: Location? = lctn
    var status: TrackerStatus = TrackerStatus.RED
    var animation = false
    var alreadyAssociated = false
    private var justUpdated = true

    private var lastUpdatedTimestamp: Long = 0
    private var animationTimeStamp: Long? = null

    private val firstDetection = det
    val trackedObjects: LinkedList<TrackedDetection> = LinkedList()
    var position = firstDetection.getCenter()
    var speed = PointF(0.0F, 0.0F)
    var speedCov: PointF = PointF(0.0f, 0.0f)

    var strength = 0.0F
    var startDate = firstDetection.timestamp

    init {
        trackedObjects.addLast(firstDetection)
    }

    fun computeMajorityClass(): String? {
        return trackedObjects.groupBy { it.classId }
                .maxByOrNull { it.value.size }?.key
    }

    fun distTo(newPos: PointF): Float {
        // Computes an adjusted distance between a point and a newPos.
        // Also anticipates the movement with speed, and computes the minimum of both distances
        val distance = malDist(position, newPos, speedCov)
        val anticipatedPosition = PointF(position.x+speed.x, position.y + speed.y)
        val anticipatedDistance = malDist(anticipatedPosition, newPos, speedCov)
        return min(distance, anticipatedDistance)
    }

    fun addDetection(newDet: TrackedDetection) {
        trackedObjects.addLast(newDet)
        position = newDet.getCenter()
        strength += 0.2F
        if(strength > 1.0F) {
            strength = 1.0F
        }
        alreadyAssociated = true
        justUpdated = true

        if(trackedObjects.size > NUM_CONSECUTIVE_DET && (status == TrackerStatus.RED)) {
            status = TrackerStatus.GREEN
            animation = true
            animationTimeStamp = newDet.timestamp
        }
    }

    fun updateSpeed(measuredSpeed: PointF, scale: Float) {
        // Move tracker directly unless it was just updated with a new detection
        if(!justUpdated) {
            position.x += measuredSpeed.x
            position.y += measuredSpeed.y
            speed = measuredSpeed
        } else {
            justUpdated = false
        }
        // Calculate the squared speed components
        val speedXSquared = speed.x * speed.x / (scale * scale)
        val speedYSquared = speed.y * speed.y / (scale * scale)

        // Update the estimated covariance using EMA
        speedCov.x =
                COVARIANCE_SMOOTHING_FACTOR * speedCov.x + (1 - COVARIANCE_SMOOTHING_FACTOR) * speedXSquared
        speedCov.y =
                COVARIANCE_SMOOTHING_FACTOR * speedCov.y + (1 - COVARIANCE_SMOOTHING_FACTOR) * speedYSquared
    }

    fun update() {
        alreadyAssociated = false

        val currTimeStamp = System.currentTimeMillis()
        val age = timeDiffInMilli(currTimeStamp, trackedObjects.last.timestamp)
        // Timber.i("AGE +> ${age.toString()} | MAX TIMESTAMP +> $MAX_TIMESTAMP")

        val ageOfAnimation = animationTimeStamp?.let {
            timeDiffInMilli(
                    currTimeStamp,
                    it
            )
        }
        if (ageOfAnimation != null) {
            if (ageOfAnimation > MAX_ANIMATION_TIMESTAMP) {
                animation = false
            }
        }

        // Green last twice as long as red
        if ((status == TrackerStatus.RED && age > MAX_TIMESTAMP) || (status == TrackerStatus.GREEN && age > MAX_TIMESTAMP * 2)) {
            status = TrackerStatus.INACTIVE
        }

        lastUpdatedTimestamp = currTimeStamp
    }

    private fun timeDiffInMilli(timestamp1: Long, timestamp2: Long): Long {
        return abs(timestamp1 - timestamp2)
    }

    fun isValid(): Boolean {
        return status == TrackerStatus.GREEN
    }

    class TrackedDetection(det: Detector.Recognition, timestamp:Long) {
        var rect: RectF = RectF(det.location)
        var detectionConfidence: Float = det.confidence
        var timestamp: Long = timestamp
        var classId: String = det.classId
        var associatedId = -1

        fun getCenter(): PointF {
            return PointF(rect.centerX(), rect.centerY())
        }
    }
    enum class TrackerStatus {
        GREEN, RED, INACTIVE
    }

    companion object {
        private const val MAX_TIMESTAMP = 2000
        private const val MAX_ANIMATION_TIMESTAMP = 1000
        private const val NUM_CONSECUTIVE_DET = 6
        private const val COVARIANCE_SMOOTHING_FACTOR = 0.2F
    }
}
