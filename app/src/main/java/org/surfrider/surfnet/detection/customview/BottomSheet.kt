package org.surfrider.surfnet.detection.customview

import android.view.View
import android.view.ViewTreeObserver
import android.widget.LinearLayout
import com.google.android.material.bottomsheet.BottomSheetBehavior
import org.surfrider.surfnet.detection.R
import org.surfrider.surfnet.detection.databinding.TfeOdActivityCameraBinding

class BottomSheet(binding: TfeOdActivityCameraBinding) {
    private var sheetBehavior: BottomSheetBehavior<LinearLayout?>? = null
    private var bottomSheetLayout = binding.bottomSheetLayout

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

                    BottomSheetBehavior.STATE_DRAGGING -> {}
                    BottomSheetBehavior.STATE_SETTLING -> bottomSheetLayout.bottomSheetArrow.setImageResource(
                        R.drawable.icn_chevron_up
                    )
                }
            }
            override fun onSlide(bottomSheet: View, slideOffset: Float) {}
        })
    }

    fun showInference(inferenceTime: String?) {
        bottomSheetLayout.inferenceInfo.text = inferenceTime
    }

    fun showGPSCoordinates(coordinates: Array<String>?) {
        if (coordinates != null && coordinates.size == 2) {
            bottomSheetLayout.latitudeInfo.text = coordinates[0]
            bottomSheetLayout.longitudeInfo.text = coordinates[1]
        } else {
            bottomSheetLayout.latitudeInfo.text = "null"
            bottomSheetLayout.longitudeInfo.text = "null"
        }
    }
}