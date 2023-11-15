package org.surfrider.surfnet.detection

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.DialogFragment
import org.surfrider.surfnet.detection.databinding.FragmentStopRecordDialogBinding
import org.surfrider.surfnet.detection.tracking.TrackerManager

class StopRecordDialog(
    private var wasteCount: Int,
    private var metersTravelled: Float,
    private var trackerManager: TrackerManager
) : DialogFragment() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setStyle(STYLE_NORMAL, R.style.CustomDialogTheme)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View? {
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
            val sendDataDialog = SendDataDialog(wasteCount, metersTravelled)
            sendDataDialog.show(parentFragmentManager, "send_data_dialog")
            dismiss() // Close the dialog
        }

        return binding.root
    }
}
