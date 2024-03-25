/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.surfrider.surfnet.detection.tflite

import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.core.Mat
import java.util.Locale

/**
 * Generic interface for interacting with different recognition engines.
 */
interface Detector {
    fun recognizeImage(bitmap: Bitmap?): List<Recognition?>?
    fun setNumThreads(numThreads: Int)

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    class Recognition(
        /**
         * class of the object
         */
        val classId: String,
        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        val confidence: Float,
        /**
         * Optional location within the source image for the location of the recognized object.
         */
        var location: RectF,

        var mask: Mat?,

        val detectedClass: Int,

        var bitmap: Bitmap?
    ) {

        override fun toString(): String {
            var resultString = "$classId "
            resultString += String.format(Locale.getDefault(), "(%.1f%%) ", confidence * 100.0f)
            resultString += "$location "

            return resultString.trim { it <= ' ' }
        }
    }
}