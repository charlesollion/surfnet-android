package org.surfrider.surfnet.detection

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.DialogFragment
import org.surfrider.surfnet.detection.databinding.FragmentStopRecordDialogBinding

class StopRecordDialog : DialogFragment() {

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

        val dialogContent = binding.dialogContent


        // Handle button click event
        binding.stopDialogNo.setOnClickListener {
            //val enteredText = binding.editText.text.toString()
            // Handle the entered text or perform any action
            // For example, you can send the text to the calling fragment
            // or perform any other logic here
            dismiss() // Close the dialog
        }

        binding.stopDialogYes.setOnClickListener {
            //val enteredText = binding.editText.text.toString()
            // Handle the entered text or perform any action
            // For example, you can send the text to the calling fragment
            // or perform any other logic here
            dismiss() // Close the dialog
        }

        return binding.root
    }
}
