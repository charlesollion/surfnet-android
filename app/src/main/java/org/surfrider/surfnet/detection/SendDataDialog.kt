package org.surfrider.surfnet.detection

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.DialogFragment
import org.surfrider.surfnet.detection.databinding.FragmentSendDataDialogBinding


class SendDataDialog(private var wasteCount: Int, private var metersTravelled: Float) : DialogFragment() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setStyle(STYLE_NORMAL, R.style.CustomDialogTheme)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View? {
        // Use DataBindingUtil to inflate the layout with data binding
        val binding = FragmentSendDataDialogBinding.inflate(
            layoutInflater
        )

        binding.sendDialogBody.text =
            getString(R.string.send_dialog_body, wasteCount.toString(), metersTravelled.toString())

        binding.dialogClearButton.setOnClickListener {
            dismiss() // Close the dialog
        }

        binding.sendDialogButton.setOnClickListener {
            activity?.finish() //Close the Camera Activity and brings back to tutorial
            dismiss() // Close the dialog
        }

        return binding.root
    }
}
