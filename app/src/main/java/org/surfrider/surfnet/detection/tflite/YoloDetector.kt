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

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.surfrider.surfnet.detection.env.Utils.loadModelFile
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
import org.tensorflow.lite.gpu.GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
import timber.log.Timber
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.util.Vector
import kotlin.math.max
import kotlin.math.pow

class YoloDetector private constructor() : Detector {
    override fun setNumThreads(numThreads: Int) {}

    // Float model
    private val IMAGE_MEAN = 0f
    private val IMAGE_STD = 255.0f

    //config yolo
    var inputSize = -1
    var numMasks = 0
    var resolutionMaskW = 0
    var resolutionMaskH = 0
    private var outputBox = 0
    private var isModelQuantized = false
    private var modelType = ""
    private var confThreshold = 0f
    private var scalingFactor = 1.0f

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = null

    // Config values.
    private val labels = Vector<String>()
    private lateinit var intValues: IntArray
    private lateinit var imgFloatArray: FloatArray
    private lateinit var imgData: ByteBuffer
    private lateinit var outData: ByteBuffer
    private lateinit var outData2: ByteBuffer
    private lateinit var masks: Mat
    private lateinit var predsCV: Mat
    private lateinit var preds: Array<FloatArray>
    private var tfLite: Interpreter? = null
    private var inp_scale = 0f
    private var inp_zero_point = 0
    private var oup_scale = 0f
    private var oup_zero_point = 0
    private var numClass = 0
    private lateinit var context: Context

    private fun convertMatToByteBuffer(mat: Mat) {
        imgData.rewind()
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        val floatMat = Mat()
        mat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / IMAGE_STD, 1.0 * IMAGE_MEAN)
        val newMat = floatMat.reshape(1, 1)
        newMat.get(0, 0, imgFloatArray)
        imgData.asFloatBuffer().put(imgFloatArray)
    }

    private fun processTFoutput(out: Mat, width:Int, height:Int): ArrayList<Recognition> {
        // Assumes a matrix of shape (output_box, 4 + numClass) or (output_box, 4 + numClass + numMask)
        val detections = ArrayList<Recognition>()
        val classes = FloatArray(numClass)
        val boxes = FloatArray(4)
        for (i in 0 until outputBox) {
            out.get(i, 4, classes)
            var detectedClass = -1
            var maxClass = 0.0f
            for (c in 0 until numClass) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                    if (maxClass > 0.5) {
                        break
                    }
                }
            }
            val confidenceInClass = maxClass
            if (confidenceInClass > confThreshold) {
                out.get(i, 0, boxes)
                val xPos = boxes[0] * scalingFactor
                val yPos = boxes[1] * scalingFactor
                val w = boxes[2] * scalingFactor
                val h = boxes[3] * scalingFactor
                val rect = RectF(
                    max(0.0f, xPos - w / 2),
                    max(0.0f, yPos - h / 2),
                    kotlin.math.min(width - 1.0f, xPos + w / 2),
                    kotlin.math.min(height - 1.0f, yPos + h / 2)
                )
                detections.add(
                    Recognition(labels[detectedClass],
                        confidenceInClass, rect, i,null, detectedClass, null
                    )
                )
            }
        }
        return detections
    }

    override fun recognizeImage(frame: Mat): ArrayList<Recognition?>? {
        return if (modelType == "segmentation") {
            segmentImage(frame)
        } else {
            detectImage(frame)
        }
    }
    private fun detectImage(frame: Mat): ArrayList<Recognition?>? {
        val preprocessTime = SystemClock.uptimeMillis()
        convertMatToByteBuffer(frame)
        val outputMap: MutableMap<Int, Any?> =
            HashMap()
        outData.rewind()
        outputMap[0] = outData
        val inputArray = arrayOf<Any?>(imgData)
        val startTime = SystemClock.uptimeMillis()
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)
        val tfTime = SystemClock.uptimeMillis()

        val byteBuffer = outputMap[0] as ByteBuffer?
        byteBuffer!!.rewind()

        // switch the way we span through the bytebuffer
        if(isModelQuantized) {
            for (j in 0 until 4 + numClass) {
                for (i in 0 until outputBox) {
                    preds[i][j] =
                        oup_scale * ((byteBuffer.get().toInt() and 0xFF) - oup_zero_point)
                }
            }
        } else {
            predsCV = Mat(4+numClass, outputBox, CvType.CV_32F, byteBuffer).t()
        }
        val allocateTime = SystemClock.uptimeMillis()
        val detections = processTFoutput(predsCV, frame.rows(), frame.cols())
        val processOutputTime = SystemClock.uptimeMillis()
        if(DEBUG_PROFILING)
            Timber.i("pre: ${startTime - preprocessTime}" +
                " tf: ${tfTime - startTime}" +
                " allocate: ${allocateTime - tfTime}" +
                " outProcess: ${processOutputTime - allocateTime}")
        return DetectorUtils.nms(detections, numClass)
    }

    private fun segmentImage(frame: Mat): ArrayList<Recognition?>? {
        val preprocessTime = SystemClock.uptimeMillis()
        convertMatToByteBuffer(frame)
        val outputMap: MutableMap<Int, Any?> =
            HashMap()
        outData.rewind()
        outData2.rewind()
        outputMap[0] = outData
        outputMap[1] = outData2
        val inputArray = arrayOf<Any?>(imgData)
        val startTime = SystemClock.uptimeMillis()
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)
        val tfTime = SystemClock.uptimeMillis()

        val byteBuffer = outputMap[0] as ByteBuffer?
        val byteBuffer2 = outputMap[1] as ByteBuffer?
        byteBuffer!!.rewind()
        byteBuffer2!!.rewind()


        // Get first output
        if(isModelQuantized) {
            for (j in 0 until 4 + numClass + numMasks) {
                for (i in 0 until outputBox) {
                    preds[i][j] =
                        oup_scale * ((byteBuffer.get().toInt() and 0xFF) - oup_zero_point)
                }
            }
        } else {
            predsCV = Mat(4+numClass+numMasks, outputBox, CvType.CV_32F, byteBuffer).t()
        }

        // Get second output, careful as it shape is HWC
        val allocateTime = SystemClock.uptimeMillis()
        masks = Mat(resolutionMaskH* resolutionMaskW, numMasks, CvType.CV_32F, byteBuffer2).t()
        val allocateMaskTime = SystemClock.uptimeMillis()
        val detections = processTFoutput(predsCV, frame.rows(), frame.cols())
        val processOutputTime = SystemClock.uptimeMillis()
        val indicesToKeep = DetectorUtils.nmsIndices(detections, numClass)
        val nmsTime = SystemClock.uptimeMillis()

        val newDetections = indicesToKeep
            .map { computeMask(detections[it], predsCV.submat(detections[it].maskIdx, detections[it].maskIdx + 1, 4+numClass, 4+numClass+numMasks))}
//        preds[detections[it].maskIdx].sliceArray(4+numClass..<4+numClass+numMasks)) }

            .toCollection(ArrayList())
        val postprocessTime = SystemClock.uptimeMillis()
        if(DEBUG_PROFILING)
            Timber.i("pre: ${startTime - preprocessTime}" +
                    " tf: ${tfTime - startTime}" +
                    " allocate: ${allocateTime - tfTime}" +
                    " allocate mask: ${allocateMaskTime - allocateTime}" +
                    " outProcess: ${processOutputTime - allocateMaskTime}" +
                    " nms: ${nmsTime - processOutputTime}" +
                    " mask: ${postprocessTime - nmsTime}")
        return newDetections
    }

    private fun computeMask(det: Recognition, maskWeights: Mat): Recognition? {
        // Performs inplace update of Mask
        val rect = Rect(det.location.left.toInt() / MASK_SCALE_FACTOR,
            det.location.top.toInt() / MASK_SCALE_FACTOR,
            det.location.width().toInt() / MASK_SCALE_FACTOR,
            det.location.height().toInt() / MASK_SCALE_FACTOR)

        det.mask = DetectorUtils.weightedSumOfMasks(masks, maskWeights, resolutionMaskH, rect)
        return det as Recognition?
    }

    companion object {
        @Throws(IOException::class)
        fun create(
            assetManager: AssetManager,
            modelFilename: String?,
            labelFilename: String,
            confThreshold: Float,
            isQuantized: Boolean,
            modelType: String,
            inputSize: Int,
            ctx: Context
        ): YoloDetector {
            val d = YoloDetector()
            d.context = ctx
            val actualFilename = labelFilename.split("file:///android_asset/".toRegex())
                .dropLastWhile { it.isEmpty() }
                .toTypedArray()[1]
            Timber.i("actual File name $actualFilename")
            val labelsInput = assetManager.open(actualFilename)
            val br = BufferedReader(InputStreamReader(labelsInput))
            var line: String?
            while (true) {
                line = br.readLine()
                if (line == null) { break } else { d.labels.add(line) }
            }
            Timber.i("First class: " + d.labels.firstElement())
            br.close()
            try {
                val compatList = CompatibilityList()
                val options = Interpreter.Options().apply{
                    if(compatList.isDelegateSupportedOnThisDevice) {
                        // if the device has a supported GPU, add the GPU delegate
                        val delegateOptions = compatList.bestOptionsForThisDevice
                        //delegateOptions.inferencePreference = INFERENCE_PREFERENCE_SUSTAINED_SPEED
                        delegateOptions.inferencePreference = INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
                        delegateOptions.setQuantizedModelsAllowed(false)
                        delegateOptions.isPrecisionLossAllowed = true
                        this.addDelegate(GpuDelegate(delegateOptions))
                    } else {
                        // if the GPU is not supported, run on 4 threads
                        Timber.i("## WARNING USING CPU ##")
                        this.useXNNPACK = true
                        this.numThreads = 4
                    }
                }
                d.tfliteModel = loadModelFile(assetManager, modelFilename)
                d.tfLite = Interpreter(d.tfliteModel!!, options)
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
            d.confThreshold = confThreshold
            d.isModelQuantized = isQuantized
            d.scalingFactor = inputSize.toFloat()
            d.modelType = modelType

            // Pre-allocate buffers.
            val numBytesPerChannel = if (isQuantized) 1 else 4
            d.inputSize = inputSize
            d.imgData =
                ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel)
            d.imgFloatArray = FloatArray(d.inputSize * d.inputSize * 3)
            d.imgData.order(ByteOrder.nativeOrder())
            d.intValues = IntArray(d.inputSize * d.inputSize)

            // yolov8 case (20² + 40² + 80²) = 8400
            d.outputBox = ((inputSize / 32).toDouble().pow(2.0) + (inputSize / 16).toDouble()
                .pow(2.0) + (inputSize / 8).toDouble().pow(2.0)).toInt()


            if (d.isModelQuantized) {
                val inpten = d.tfLite!!.getInputTensor(0)
                d.inp_scale = inpten.quantizationParams().scale
                d.inp_zero_point = inpten.quantizationParams().zeroPoint
                val oupten = d.tfLite!!.getOutputTensor(0)
                d.oup_scale = oupten.quantizationParams().scale
                d.oup_zero_point = oupten.quantizationParams().zeroPoint
            }
            val shape = d.tfLite!!.getOutputTensor(0).shape()

            val numClass: Int
            when (modelType) {
                "detection" -> {
                    numClass = shape[shape.size - 2] - 4
                    d.outData =
                        ByteBuffer.allocateDirect(d.outputBox * (numClass + 4) * numBytesPerChannel)
                    d.preds = Array(d.outputBox) { FloatArray(shape[shape.size - 2]) }
                }
                "segmentation" -> {
                    // segmentation case (1, num_class+ num_masks+4, num_anchors)
                    val shape2 = d.tfLite!!.getOutputTensor(1).shape()
                    d.resolutionMaskW = shape2[1]
                    d.resolutionMaskH = shape2[2]
                    d.numMasks = shape2[3]
                    numClass = shape[1] - 4 - d.numMasks
                    d.outData =
                        ByteBuffer.allocateDirect(d.outputBox * (4 + numClass + d.numMasks) * numBytesPerChannel)
                    // second output: masks (1, H, W, num_masks)
                    d.outData2 =
                        ByteBuffer.allocateDirect((d.resolutionMaskW * d.resolutionMaskH * d.numMasks)* numBytesPerChannel)
                    // d.masks = Mat(d.numMasks, d.resolutionMaskW * d.resolutionMaskH, CvType.CV_32F)
                    d.outData2.order(ByteOrder.nativeOrder())
                    d.preds = Array(d.outputBox) { FloatArray(shape[1]) }
                }
                else -> {
                    throw RuntimeException("model type $modelType not recognized, possible types: [detection, segmentation]")
                }
            }
            d.numClass = numClass
            d.outData.order(ByteOrder.nativeOrder())

            return d
        }
        private const val MASK_SCALE_FACTOR = 4
        private const val DEBUG_PROFILING = true
    }
}