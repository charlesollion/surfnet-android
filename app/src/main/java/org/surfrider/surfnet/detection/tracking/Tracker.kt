package org.surfrider.surfnet.detection.tracking

import android.graphics.PointF
import android.graphics.RectF
import org.opencv.core.Core
import org.opencv.core.Core.*
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.times
import org.opencv.video.KalmanFilter
import org.surfrider.surfnet.detection.tflite.Detector
import timber.log.Timber
import java.util.*
import kotlin.math.abs

public class Tracker(det: TrackedDetection, idx: Int, dt: Double) {
    private val MAX_TIMESTAMP = 3000
    private val MAX_ANIMATION_TIMESTAMP = 1000
    private val NUM_CONSECUTIVE_DET = 5

    private var kalmanFilter: KalmanFilter
    private var state: Mat

    var index = idx
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
        kalmanFilter = KalmanFilter(4, 4, 0, CvType.CV_32F)
        // Initialize with first position, and 0 speed
        state = Mat.zeros(4, 1, CvType.CV_32F)
        state.put(0, 0, position.x.toDouble())
        state.put(1, 0, position.y.toDouble())

        // Define transition matrix (A) - movement equation
        val A = Mat.eye(4, 4, CvType.CV_32F)
        A.put(0, 2, dt)
        A.put(1, 3, dt)
        kalmanFilter._transitionMatrix = A

        // Define measurement matrix (H) - observe position and velocity
        kalmanFilter._measurementMatrix = Mat.eye(4, 4, CvType.CV_32F)

        // Define process noise covariance (Q) - tune this based on your problem
        val Q = Mat(4, 4, CvType.CV_32F)
        setIdentity(Q, Scalar(1.0))
        kalmanFilter._processNoiseCov = Q

        // Define measurement noise covariance (R) - tune this based on your problem
        val R = Mat(4, 4, CvType.CV_32F)
        setIdentity(R, Scalar(1.0))
        kalmanFilter._measurementNoiseCov = R

        state.copyTo(kalmanFilter._statePre)

        /*val newPosition = firstDetection.getCenter()
        val measurement = Mat.zeros(4, 1, CvType.CV_32F)
        measurement.put(0, 0, newPosition.x.toDouble())
        measurement.put(1, 0, newPosition.y.toDouble())

        // Set the velocity components in the state vector to their current values
        measurement.put(2, 0, state.get(2, 0)[0]) // Keep the current vx
        measurement.put(3, 0, state.get(3, 0)[0]) // Keep the current vy

        // Update the state with the measurement
        kalmanFilter.correct(measurement)*/

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
        val newPosition = newDet.getCenter()
        val measurement = Mat.zeros(4, 1, CvType.CV_32F)
        measurement.put(0, 0, newPosition.x.toDouble())
        measurement.put(1, 0, newPosition.y.toDouble())

        // Set the velocity components in the state vector to their current values
        measurement.put(2, 0, state.get(2, 0)[0]) // Keep the current vx
        measurement.put(3, 0, state.get(3, 0)[0]) // Keep the current vy

        // Update the state with the measurement
        kalmanFilter.correct(measurement)
        kalmanFilter.predict()

        alreadyAssociated = true

        if(trackedObjects.size > NUM_CONSECUTIVE_DET && status == TrackerStatus.RED) {
            status = TrackerStatus.GREEN
            animation = true
            animationTimeStamp = newDet.timestamp
        }
    }

    fun updateSpeed(measuredSpeed: PointF) {
        // Create a measurement vector
        val measurement = Mat.zeros(4, 1, CvType.CV_32F)

        measurement.put(0, 0, state.get(0, 0)[0]) // Keep the current x
        measurement.put(1, 0, state.get(1, 0)[0]) // Keep the current y

        // Set the velocity components in the state vector to their current values
        measurement.put(2, 0, measuredSpeed.x.toDouble()) // Keep the current vx
        measurement.put(3, 0, measuredSpeed.y.toDouble()) // Keep the current vy

        Timber.i("#${index} - Updating speed: ${measurement.dump()}")

        // Update the state with the measurement
        kalmanFilter.correct(measurement)
    }



    fun update() {
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
            val dt = compareTimeDifferenceInMilliseconds(currTimeStamp, lastUpdatedTimestamp) / 1000.0
            val A = Mat.eye(4, 4, CvType.CV_32F)
            A.put(0, 2, dt)
            A.put(1, 3, dt)
            kalmanFilter._transitionMatrix = A
            val outputPredict = kalmanFilter.predict()
            outputPredict.copyTo(state)
            Timber.i("#${index} - output of predict: ${outputPredict.dump()}")


            position.x = state.get(0, 0)[0].toFloat() // Estimated x
            position.y = state.get(1, 0)[0].toFloat() // Estimated x
            speed.x = state.get(2, 0)[0].toFloat() // Estimated vx
            speed.y = state.get(3, 0)[0].toFloat() // Estimated vy
            // Timber.i("Updating state: ${position} $speed")
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
