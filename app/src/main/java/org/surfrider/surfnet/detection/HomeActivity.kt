package org.surfrider.surfnet.detection

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.text.SpannableStringBuilder
import androidx.appcompat.app.AppCompatActivity
import org.surfrider.surfnet.detection.databinding.ActivityHomeBinding


class HomeActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHomeBinding



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val sharedPreference =  getSharedPreferences("email", Context.MODE_PRIVATE)

        binding.editTextTextEmailAddress.text= SpannableStringBuilder(sharedPreference.getString("email",""))


        binding.button.setOnClickListener {

            var editor = sharedPreference.edit()
            editor.putString("email",binding.editTextTextEmailAddress.text.toString())
            editor.commit()
            val intent = Intent(this, DetectorActivity::class.java)
            startActivity(intent)
        }
    }


}