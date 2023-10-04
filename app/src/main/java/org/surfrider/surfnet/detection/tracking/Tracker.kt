package org.surfrider.surfnet.detection.tracking

import android.graphics.Point
import android.graphics.PointF
import android.graphics.RectF
import org.surfrider.surfnet.detection.tflite.Detector
import java.util.*

public class Tracker(det: TrackedDetection, idx: Int) {
    private val MAX_TIMESTAMP = 1000
    private val NUM_CONSECUTIVE_DET = 5
    private val ASSOCIATION_THRESHOLD = 10.0F

    var index = idx
    var status : String = "red"
    var associated = false

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
        associated = true

        if(trackedObjects.size > NUM_CONSECUTIVE_DET) {
            status = "green"
        }
    }

    fun update(elapsedTime: Int) {
        for(trackedObject in trackedObjects) {
            trackedObject.timestamp += elapsedTime
        }
        if(trackedObjects.last.timestamp > MAX_TIMESTAMP) {
            status = "inactive"
        }
    }

    class TrackedDetection(det: Detector.Recognition) {
        var location: RectF = RectF(det.location)
        var detectionConfidence = det.confidence
        var timestamp: Int = 0
        var classId: String = det.id
        var associatedId = -1

        fun getCenter(): PointF {
            return PointF(location.centerX(), location.centerY())
        }
    }
}
