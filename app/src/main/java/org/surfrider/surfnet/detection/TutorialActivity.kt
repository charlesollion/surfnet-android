package org.surfrider.surfnet.detection

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import org.surfrider.surfnet.detection.databinding.ActivityTutorialBinding

class TutorialActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTutorialBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityTutorialBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.readyButton.setOnClickListener {
            val intent = Intent(this, DetectorActivity::class.java)
            startActivity(intent)
        }
    }
}