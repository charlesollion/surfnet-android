package org.surfrider.surfnet.detection.env

import android.content.res.AssetManager
import java.io.*
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

object Utils {
    /**
     * Memory-map the model file in Assets.
     */
    @JvmStatic
    @Throws(IOException::class)
    fun loadModelFile(assets: AssetManager, modelFilename: String?): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename!!)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }



}