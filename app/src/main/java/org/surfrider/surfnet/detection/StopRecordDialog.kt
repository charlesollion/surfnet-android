package org.surfrider.surfnet.detection

import android.content.Context
import android.location.Location
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.DialogFragment
import org.surfrider.surfnet.detection.databinding.FragmentStopRecordDialogBinding
import org.surfrider.surfnet.detection.tracking.TrackerManager

class StopRecordDialog(
    private var trackerManager: TrackerManager
) : DialogFragment() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setStyle(STYLE_NORMAL, R.style.CustomDialogTheme)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        // Use DataBindingUtil to inflate the layout with data binding
        val binding = FragmentStopRecordDialogBinding.inflate(
            layoutInflater
        )

        // Handle button click event
        binding.stopDialogNo.setOnClickListener {

            dismiss() // Close the dialog
        }

        binding.stopDialogYes.setOnClickListener {

            context?.let {
                val sharedPreference = it.getSharedPreferences("prefs", Context.MODE_PRIVATE)
                trackerManager.sendData(
                    it,
                    sharedPreference?.getString("email", "null")
                )
            }
            //Calculate the distance the user made
            var distance = 0F
            if (trackerManager.positions.size >= 2) {
                for (i in 0 until trackerManager.positions.size - 1) {
                    val results = FloatArray(1)
                    Location.distanceBetween(
                        trackerManager.positions[i].lat,
                        trackerManager.positions[i].lng,
                        trackerManager.positions[i + 1].lat,
                        trackerManager.positions[i + 1].lng,
                        results
                    )
                    distance += results[0]
                }
            }

            val sendDataDialog =
                SentDataDialog(
                    wasteCount = trackerManager.detectedWaste.size,
                    metersTravelled = distance
                )
            sendDataDialog.show(parentFragmentManager, "send_data_dialog")
            dismiss() // Close the dialog
        }

        return binding.root
    }
}
