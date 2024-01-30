package org.surfrider.surfnet.detection.customview

import android.content.Context
import android.view.View
import android.view.ViewTreeObserver
import android.widget.LinearLayout
import com.google.android.material.bottomsheet.BottomSheetBehavior
import org.surfrider.surfnet.detection.R
import org.surfrider.surfnet.detection.databinding.ActivityCameraBinding

import java.text.DecimalFormat
import org.surfrider.surfnet.detection.tracking.TrackerManager


class BottomSheet(binding: ActivityCameraBinding) {
    private var sheetBehavior: BottomSheetBehavior<LinearLayout?>? = null
    private var bottomSheetLayout = binding.bottomSheetLayout
    private var df = DecimalFormat("#.##")
    private var switchOF = bottomSheetLayout.switch1
    private var switchBoxes = bottomSheetLayout.switch2
    var showOF = true
    var showBoxes = false

    init {
        val bottomSheetLayout = binding.bottomSheetLayout
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout.bottomSheetLayout)
        val vto = bottomSheetLayout.gestureLayout.viewTreeObserver
        vto.addOnGlobalLayoutListener(object : ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                bottomSheetLayout.gestureLayout.viewTreeObserver.removeOnGlobalLayoutListener(
                    this
                )
                val height = bottomSheetLayout.gestureLayout.measuredHeight
                sheetBehavior!!.peekHeight = height
            }
        })
        sheetBehavior?.isHideable = false
        sheetBehavior?.addBottomSheetCallback(object : BottomSheetBehavior.BottomSheetCallback() {
            override fun onStateChanged(bottomSheet: View, newState: Int) {
                when (newState) {
                    BottomSheetBehavior.STATE_HIDDEN -> {}
                    BottomSheetBehavior.STATE_EXPANDED -> {
                        bottomSheetLayout.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_down)
                    }

                    BottomSheetBehavior.STATE_COLLAPSED -> {
                        bottomSheetLayout.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_up)
                    }

                    BottomSheetBehavior.STATE_HALF_EXPANDED -> {}
                    BottomSheetBehavior.STATE_DRAGGING -> {}
                    BottomSheetBehavior.STATE_SETTLING -> bottomSheetLayout.bottomSheetArrow.setImageResource(
                        R.drawable.icn_chevron_up
                    )
                }
            }

            override fun onSlide(bottomSheet: View, slideOffset: Float) {}
        })
        bottomSheetLayout.materialSwitch.isChecked = true

        switchOF.setOnCheckedChangeListener { _, isChecked ->
            showOF = isChecked
        }
        switchBoxes.setOnCheckedChangeListener { _, isChecked ->
            showBoxes = isChecked
        }
    }

    fun showInference(inferenceTime: String?) {
        bottomSheetLayout.inferenceInfo.text = inferenceTime
    }

    fun showGPSCoordinates(coordinates: Array<String>?) {
        if (coordinates != null && coordinates.size == 2) {
            bottomSheetLayout.latitudeInfo.text = coordinates[0]
            bottomSheetLayout.longitudeInfo.text = coordinates[1]
        } else {
            bottomSheetLayout.latitudeInfo.text = ""
            bottomSheetLayout.longitudeInfo.text = ""
        }
    }

    fun showIMUStats(context: Context, stats: Array<Float>?) {
        if (stats != null && stats.size == 6) {
            bottomSheetLayout.positionInfo.text = context.getString(
                R.string.debug_three_vars,
                df.format(stats[0]),
                df.format(stats[1]),
                df.format(stats[2])
            )
            bottomSheetLayout.speedInfo.text = df.format(stats[3])
            bottomSheetLayout.flowInfo.text =
                context.getString(R.string.debug_two_vars, df.format(stats[4]), df.format(stats[5]))
        } else {
            bottomSheetLayout.positionInfo.text = ""
            bottomSheetLayout.speedInfo.text = ""
            bottomSheetLayout.flowInfo.text = ""
        }
    }

    fun displayDetection(tracker: TrackerManager) {
        bottomSheetLayout.materialSwitch.setOnCheckedChangeListener { _, isChecked ->
            tracker.displayDetection = isChecked
        }
    }

}