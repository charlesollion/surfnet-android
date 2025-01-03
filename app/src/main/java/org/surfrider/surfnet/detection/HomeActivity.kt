package org.surfrider.surfnet.detection

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.method.LinkMovementMethod
import android.text.style.ClickableSpan
import android.util.Patterns
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
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
            if (validateEmail()) {
                val editor = sharedPreference.edit()
                editor.putString("email", binding.editTextTextEmailAddress.text.toString())
                editor.apply()
                val intent = Intent(this, TutorialActivity::class.java)
                startActivity(intent)
            }
        }

        setupPermissions()

        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
    }

    private fun validateEmail(): Boolean {
        val email = binding.editTextTextEmailAddress.text.toString().trim()
        val pattern = Patterns.EMAIL_ADDRESS
        return if (pattern.matcher(email).matches()) {
            true
        } else {
            binding.editTextTextEmailAddress.error = "Vous devez saisir une adresse email valide"
            false
        }
    }

    private fun setupPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_FINE_LOCATION
        )
        if (!checkPermissions(permissions)) {
            // TODO add dialog for permissions
            //val locationPermissionDialog = LocationPermissionDialog()
            //locationPermissionDialog.show(supportFragmentManager, "stop_record_dialog")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(permissions, 1)
            }
        }
    }
    private fun checkPermissions(permissions: Array<String>): Boolean {
        for (permission in permissions) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    permission
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }

}