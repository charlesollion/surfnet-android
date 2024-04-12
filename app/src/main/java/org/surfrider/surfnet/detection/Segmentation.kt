package org.surfrider.surfnet.detection

import android.app.Application
import timber.log.Timber

class Segmentation: Application() {

    override fun onCreate() {
        super.onCreate()
        if(BuildConfig.DEBUG){
            Timber.plant(Timber.DebugTree())
        }
        instance = this
    }

    companion object {
        lateinit var instance: Segmentation
            private set
    }
}