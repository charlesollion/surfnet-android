package org.surfrider.surfnet.detection

import android.app.Application
import timber.log.Timber

class Surfrider: Application() {

    override fun onCreate() {
        super.onCreate()
        if(BuildConfig.DEBUG){
            Timber.plant(Timber.DebugTree())
        }
        instance = this
    }

    companion object {
        lateinit var instance: Surfrider
            private set
    }
}