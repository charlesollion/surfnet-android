package org.surfrider.surfnet.detection.tracking

import android.content.Context
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.VectorDrawable
import android.location.Location
import androidx.core.content.ContextCompat
import org.surfrider.surfnet.detection.R
import org.surfrider.surfnet.detection.model.TrackerResult
import org.surfrider.surfnet.detection.model.TrackerTrash
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import timber.log.Timber
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.min

class TrackerManager {
    val trackers: LinkedList<Tracker> = LinkedList<Tracker>()
    private var trackerIndex = 0
    val detectedWaste: LinkedList<Tracker> = LinkedList<Tracker>()


    private fun updateTrackers() {
        trackers.forEach { tracker -> tracker.update() }
    }

    @Synchronized
    fun processDetections(results: List<Recognition>, location: Location?) {
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
                if (tracker.status != Tracker.TrackerStatus.INACTIVE && !tracker.alreadyAssociated) {
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
                trackers.addLast(Tracker(det, trackerIndex, location))
                trackerIndex++
            }
        }
    }

    @Synchronized
    fun draw(canvas: Canvas, context: Context?, previewWidth: Int, previewHeight: Int) {
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
                        it, if (tracker.status == Tracker.TrackerStatus.GREEN) {
                            if (!detectedWaste.contains(tracker)) {
                                detectedWaste.add(tracker)
                            }
                            R.drawable.green_dot
                        } else {
                            R.drawable.red_dot
                        }
                    )
                }



                if (bmp != null) {
                    val bmpWidth = bmp.width.div(scale)
                    val bmpHeight = bmp.height.div(scale)

                    val point =
                        floatArrayOf(trackedPos.x - bmpWidth / 2, trackedPos.y - bmpHeight / 2)
                    frameToCanvasTransform.mapPoints(point)

                    canvas.drawBitmap(bmp, point[0], point[1], null)

                    //affichage du text avec le numéro du tracker
                    val paint = Paint()
                    paint.textSize = 40.0F
                    canvas.drawText(tracker.index.toString(), point[0], point[1], paint)


                    //Animation drawing
                    if (tracker.animation) {
                        val shouldShowBottomAnimation = trackedPos.y < canvas.height.div(scale) / 2
                        val animation = getBitmap(
                            context,
                            if (shouldShowBottomAnimation) R.drawable.animation_down else R.drawable.animation
                        )

                        if (animation != null) {
                            val animationWidth = animation.width.div(scale)
                            val animationHeight = animation.height.div(scale)

                            val animationPoint = floatArrayOf(
                                trackedPos.x - (animationWidth / 2) + 3,
                                if (shouldShowBottomAnimation) trackedPos.y + bmpHeight / 2 else trackedPos.y - bmpHeight / 2 - (animationHeight)
                            )
                            frameToCanvasTransform.mapPoints(animationPoint)
                            canvas.drawBitmap(animation, animationPoint[0], animationPoint[1], null)
                        }
                    }
                }
            }
        }
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

    fun sendData(email: String) {
        var trashes = ArrayList<TrackerTrash>()
        trackers.forEach {
            val date = Date(it.startDate)
            val iso8601Format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'")
            val iso8601DateString = iso8601Format.format(date)
            Timber.i(iso8601DateString)
            trashes.add(
                TrackerTrash(
                    date = iso8601DateString,
                    lat = it.location?.latitude,
                    lng = it.location?.longitude,
                    name = "unknown"
                )
            )
        }

        var result = TrackerResult(
            move = null,
            bank = null,
            trackingMode = "automatic",
            files = ArrayList(),
            trashes = trashes,
            positions = ArrayList(),
            comment = "email : $email"
        )
    }
}