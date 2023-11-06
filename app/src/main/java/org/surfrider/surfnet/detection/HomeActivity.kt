package org.surfrider.surfnet.detection

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.method.LinkMovementMethod
import android.text.style.ClickableSpan
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import org.surfrider.surfnet.detection.databinding.ActivityHomeBinding


class HomeActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHomeBinding



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val cguView = binding.cgu

        // Create a SpannableString
        val cguText = resources.getString(R.string.home_cgu)

        val spannableString = SpannableString(cguText)

        // Create a ClickableSpan
        val clickableSpan = object : ClickableSpan() {
            override fun onClick(view: View) {
                val browserIntent = Intent(Intent.ACTION_VIEW, Uri.parse("https://www.plasticorigins.eu/CGU"))
                startActivity(browserIntent)
            }
        }

        // Set the ClickableSpan on the portion of text you want to make clickable
        spannableString.setSpan(clickableSpan, 63, 66, SpannableString.SPAN_EXCLUSIVE_EXCLUSIVE)

        // Set the SpannableString to the TextView
        cguView.text = spannableString

        // Make the link clickable
        cguView.movementMethod = LinkMovementMethod.getInstance()

        val sharedPreference =  getSharedPreferences("prefs", Context.MODE_PRIVATE)

        binding.editTextTextEmailAddress.text= SpannableStringBuilder(sharedPreference.getString("email",""))


        binding.button.setOnClickListener {

            var editor = sharedPreference.edit()
            editor.putString("email",binding.editTextTextEmailAddress.text.toString())
            editor.commit()
            val intent = Intent(this, TutorialActivity::class.java)
            startActivity(intent)
        }
    }


}