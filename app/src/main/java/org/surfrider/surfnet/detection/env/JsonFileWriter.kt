package org.surfrider.surfnet.detection.env

import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.google.gson.Gson
import org.surfrider.surfnet.detection.models.TrackerResult
import timber.log.Timber
import java.io.File
import java.io.FileWriter
import java.io.OutputStream


class JsonFileWriter {

    companion object {

        private const val TAG = "JsonFileWriter"

        fun writeResultToJsonFile(context: Context, result: TrackerResult) {
            val gson = Gson()
            val jsonString = gson.toJson(result)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Use MediaStore to write to the Downloads folder
                val resolver: ContentResolver = context.contentResolver
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, "trace.json")
                    put(MediaStore.MediaColumns.MIME_TYPE, "application/json")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }

                val contentUri = MediaStore.Downloads.EXTERNAL_CONTENT_URI
                var outputStream: OutputStream? = null

                try {
                    val contentUri = resolver.insert(contentUri, contentValues)
                    contentUri?.let { uri ->
                        outputStream = resolver.openOutputStream(uri)
                        outputStream?.use { it.write(jsonString.toByteArray()) }
                        Timber.tag(TAG).i("JSON written to Downloads folder successfully")
                    }
                } catch (e: Exception) {
                    Timber.tag(TAG).e("Error writing JSON to Downloads folder: ${e.message}")
                } finally {
                    outputStream?.close()
                }
            } else {
                try {
                    val dir =
                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    val file = File(dir, "output.json")
                    val writer = FileWriter(file)
                    writer.use {
                        it.write(jsonString)
                    }
                    Timber.tag(TAG).i("JSON written to file successfully")
                } catch (e: Exception) {
                    Timber.tag(TAG).e("Error writing JSON to file: ${e.message}")
                }
            }
        }
    }
}