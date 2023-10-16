package org.surfrider.surfnet.detection.tracking

import android.content.Context
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.VectorDrawable
import androidx.core.content.ContextCompat
import org.surfrider.surfnet.detection.R
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import timber.log.Timber
import java.util.*

class TrackerManager {
    val trackers: LinkedList<Tracker> = LinkedList<Tracker>()
    private var trackerIndex = 0
    var detectedWaste = 0

    fun updateTrackers() {
        trackers.forEach { tracker -> tracker.update() }
    }

    @Synchronized
    fun processDetections(results: List<Recognition>) {
        updateTrackers()
        // Store all Recognition objects in a list of TrackedDetections
        val dets = LinkedList<Tracker.TrackedDetection>()
        for (result in results) {
            dets.addLast(Tracker.TrackedDetection(result))
        }

        // Associate results with current trackers
        for (det in dets) {
            val position = det.getCenter()
            var minDist = 10000.0F
            // Timber.i("Trackers Size = ${trackers.size}")
            // Greedy assignment of trackers
            trackers.forEachIndexed { i, tracker ->
                if(tracker.status != Tracker.TrackerStatus.INACTIVE && !tracker.alreadyAssociated) {
                    val dist = tracker.distTo(position)
                    // Timber.i("Distance = $dist")
                    if (dist < minDist) {
                        minDist = dist
                        det.associatedId = i
                    }
                }

            }
            if (det.associatedId != -1) {
                trackers[det.associatedId].addDetection(det)
            } else {
                trackers.addLast(Tracker(det, trackerIndex))
                trackerIndex++
            }
        }
    }
    @Synchronized
    fun draw(canvas: Canvas, context: Context?, previewWidth:Int, previewHeight:Int) {
        // Build transform matrix from canvas and context
        val frameToCanvasTransform = Matrix()
        val scale = Math.min(canvas!!.width / previewWidth.toFloat(),
                             canvas!!.height / previewHeight.toFloat())
        frameToCanvasTransform.postScale(scale, scale)
            var i = 0

        for (tracker in trackers) {
            val trackedPos = tracker.position
            //Only draw tracker if not inactive
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                val bmp = context?.let {
                    getBitmap(
                        it,
                        if (tracker.status == Tracker.TrackerStatus.GREEN) {
                            i++
                            R.drawable.green_dot
                        } else {
                            R.drawable.red_dot
                        }
                    )
                }
                val point = floatArrayOf(trackedPos.x, trackedPos.y)
                frameToCanvasTransform.mapPoints(point)
                if (bmp != null) {
                    canvas.drawBitmap(bmp, point[0], point[1], null)
                }

                //affichage du text avec le numéro du tracker
                val paint = Paint()
                paint.textSize = 40.0F
                canvas.drawText(tracker.index.toString(), point[0], point[1], paint)
            }
        }
        detectedWaste = i;
    }

    @Synchronized
    fun drawDebug(canvas: Canvas) {
        for (tracker in trackers) {
            val trackedPos = tracker.position
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                val paint = Paint()
                paint.textSize = 40.0F
                canvas.drawText(tracker.index.toString(), trackedPos.x, trackedPos.y,paint )
            }
        }
    }

    private fun getBitmap(vectorDrawable: VectorDrawable?): Bitmap? {
        vectorDrawable?.let {
            val bitmap = Bitmap.createBitmap(
                it.intrinsicWidth * 2,
                it.intrinsicHeight * 2,
                Bitmap.Config.ARGB_8888
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

}