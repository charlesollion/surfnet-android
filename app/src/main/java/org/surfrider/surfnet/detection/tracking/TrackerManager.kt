package org.surfrider.surfnet.detection.tracking

import android.annotation.TargetApi
import android.content.Context
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.VectorDrawable
import android.os.Build
import androidx.core.content.ContextCompat
import org.surfrider.surfnet.detection.R
import org.surfrider.surfnet.detection.env.ImageUtils.getTransformationMatrix
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import timber.log.Timber
import java.util.*
import kotlin.math.min

class TrackerManager {
    val trackers: LinkedList<Tracker> = LinkedList<Tracker>()
    var trackerIndex = 0
    private var frameToCanvasMatrix: Matrix? = null
    private var frameWidth = 0
    private var frameHeight = 0
    private var sensorOrientation = 0

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
            Timber.i("Trackers Size = ${trackers.size}")
            // Greedy assignment of trackers
            trackers.forEachIndexed { i, tracker ->
                if(tracker.status != Tracker.TrackerStatus.INACTIVE && !tracker.alreadyAssociated) {
                    val dist = tracker.distTo(position)
                    Timber.i("Distance = ${dist}")
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


    //TODO Refactor all old code under this todo

    @Synchronized
    fun setFrameConfiguration(width: Int, height: Int, sensorOrientation: Int) {
        frameWidth = width
        frameHeight = height
        this.sensorOrientation = sensorOrientation
    }

    @Synchronized
    fun draw(canvas: Canvas, context: Context?) {
        val rotated = sensorOrientation % 180 == 90
        val multiplier: Float = min(
            canvas.height / (if (rotated) frameWidth else frameHeight).toFloat(),
            canvas.width / (if (rotated) frameHeight else frameWidth).toFloat()
        )
        frameToCanvasMatrix = getTransformationMatrix(
            frameWidth,
            frameHeight,
            (multiplier * if (rotated) frameHeight else frameWidth).toInt(),
            (multiplier * if (rotated) frameWidth else frameHeight).toInt(),
            sensorOrientation,
            false
        )
        for (tracker in trackers) {
            val trackedPos = tracker.position
            //Only draw tracker if not inactive
            if (tracker.status != Tracker.TrackerStatus.INACTIVE) {
                val bmp = context?.let {
                    getBitmap(
                        it,
                        if (tracker.status == Tracker.TrackerStatus.GREEN) R.drawable.green_dot else R.drawable.red_dot
                    )
                }
                if (bmp != null) {
                    canvas.drawBitmap(bmp, trackedPos.x, trackedPos.y, null)
                }

                //affichage du text avec le numéro du tracker
                val paint = Paint()
                paint.textSize = 40.0F
                canvas.drawText(tracker.index.toString(), trackedPos.x, trackedPos.y,paint )


                /*
            Old boxes +>

            boxPaint.setColor(recognition.color);
            float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
            canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);
            final String labelString =
                  !TextUtils.isEmpty(recognition.title)
                      ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
                      : String.format("%.2f", (100 * recognition.detectionConfidence));
                  borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
            labelString);
            borderedText.drawText(
                   canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
            */
            }
        }
    }

    @TargetApi(Build.VERSION_CODES.LOLLIPOP)
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
        val drawable = ContextCompat.getDrawable(context, drawableId)
        return if (drawable is BitmapDrawable) {
            BitmapFactory.decodeResource(context.resources, drawableId)
        } else if (drawable is VectorDrawable) {
            getBitmap(drawable as VectorDrawable?)
        } else {
            throw IllegalArgumentException("unsupported drawable type")
        }
    }

}