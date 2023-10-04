package org.surfrider.surfnet.detection.tracking

import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import java.util.LinkedList

class TrackerManager {
    val trackers : LinkedList<Tracker> = LinkedList<Tracker>()
    var trackerIndex = 0

    fun processDetections(results: List<Recognition>) {
        // Store all Recognition objects in a list of TrackedDetections
        val dets = LinkedList<Tracker.TrackedDetection>()
        for(result in results) {
            dets.addLast(Tracker.TrackedDetection(result))
        }

        // Associate results with current trackers
        for(det in dets) {
            val position = det.getCenter()
            var minDist = 10000.0F

            // Greedy assignment of trackers
            trackers.forEachIndexed {i, tracker ->
                if(tracker.associated == false) {
                    val dist = tracker.distTo(position)
                    if(dist < minDist) {
                        minDist = dist
                        det.associatedId = i
                    }
                }
            }
            trackers[det.associatedId].addDetection(det)
        }

        // For each result without association
        for(det in dets) {
            if(det.associatedId == -1) {
                trackers.addLast(Tracker(det, trackerIndex))
                trackerIndex++
            }
        }
    }

}