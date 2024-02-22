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

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import org.surfrider.surfnet.detection.env.Utils.loadModelFile
import org.surfrider.surfnet.detection.tflite.Detector.Recognition
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
import org.tensorflow.lite.gpu.GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
import org.tensorflow.lite.nnapi.NnApiDelegate
import timber.log.Timber
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.lang.Math.min
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.PriorityQueue
import java.util.Vector
import kotlin.math.max

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
class YoloDetector private constructor() : Detector {
    override fun setNumThreads(num_threads: Int) {}
    private fun recreateInterpreter() {
        if (tfLite != null) {
            tfLite!!.close()
            tfLite = Interpreter(tfliteModel!!, tfliteOptions)
        }
    }

    fun useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = GpuDelegate()
            tfliteOptions.addDelegate(gpuDelegate)
            recreateInterpreter()
        }
    }

    fun useCPU() {
        recreateInterpreter()
    }

    fun useNNAPI() {
        nnapiDelegate = NnApiDelegate()
        tfliteOptions.addDelegate(nnapiDelegate)
        recreateInterpreter()
    }

    // Float model
    private val IMAGE_MEAN = 0f
    private val IMAGE_STD = 255.0f

    //config yolo
    var inputSize = -1
    private var output_box = 0
    private var isModelQuantized = false
    private var isV8 = false
    private var confThreshold = 0f

    /** holds a gpu delegate  */
    var gpuDelegate: GpuDelegate? = null

    /** holds an nnapi delegate  */
    var nnapiDelegate: NnApiDelegate? = null

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = null

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    // Config values.
    private val labels = Vector<String>()
    private lateinit var intValues: IntArray
    private lateinit var imgData: ByteBuffer
    private lateinit var outData: ByteBuffer
    private var tfLite: Interpreter? = null
    private var inp_scale = 0f
    private var inp_zero_point = 0
    private var oup_scale = 0f
    private var oup_zero_point = 0
    private var numClass = 0

    //non maximum suppression
    protected fun nms(list: ArrayList<Recognition>): ArrayList<Recognition?> {
        val nmsList = ArrayList<Recognition?>()
        for (k in labels.indices) {
            //1.find max confidence per class
            val pq = PriorityQueue<Recognition?>(
                50
            ) { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                java.lang.Float.compare(rhs.confidence!!, lhs.confidence!!)
            }
            for (i in list.indices) {
                if (list[i].detectedClass == k) {
                    pq.add(list[i])
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a = arrayOfNulls<Recognition>(pq.size)
                val detections = pq.toArray(a)
                val max = detections[0]
                nmsList.add(max)
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection = detections[j]
                    val b = detection!!.location
                    if (box_iou(max!!.location, b) < mNmsThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsList
    }

    protected var mNmsThresh = 0.6f
    protected fun box_iou(a: RectF, b: RectF): Float {
        return box_intersection(a, b) / box_union(a, b)
    }

    protected fun box_intersection(a: RectF, b: RectF): Float {
        val w = overlap(
            (a.left + a.right) / 2, a.right - a.left,
            (b.left + b.right) / 2, b.right - b.left
        )
        val h = overlap(
            (a.top + a.bottom) / 2, a.bottom - a.top,
            (b.top + b.bottom) / 2, b.bottom - b.top
        )
        return if (w < 0 || h < 0) 0.0F else w * h
    }

    protected fun box_union(a: RectF, b: RectF): Float {
        val i = box_intersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    protected fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = if (l1 > l2) l1 else l2
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = if (r1 < r2) r1 else r2
        return right - left
    }

    /**
     * Writes Image data into a `ByteBuffer`.
     */
    protected fun convertBitmapToByteBuffer(bitmap: Bitmap?): ByteBuffer? {
        bitmap!!.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val pixel = 0
        imgData!!.rewind()
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                if (isModelQuantized) {
                    // Quantized model
                    imgData!!.put((((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
                    imgData!!.put((((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
                    imgData!!.put((((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
                } else { // Float model
                    imgData!!.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData!!.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData!!.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }
        return imgData
    }

    override fun recognizeImage(bitmap: Bitmap?): ArrayList<Recognition?>? {
        convertBitmapToByteBuffer(bitmap)
        val outputMap: MutableMap<Int, Any?> =
            HashMap()
        outData!!.rewind()
        outputMap[0] = outData
        // Timber.tag("YoloDetector").d("mObjThresh: %s", getObjThresh());
        val inputArray = arrayOf<Any?>(imgData)
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)
        val byteBuffer = outputMap[0] as ByteBuffer?
        byteBuffer!!.rewind()
        val detections = ArrayList<Recognition>()
        val out =
            Array(1) {
                Array(output_box) { FloatArray(numClass + 5) }
            }
        if (!isV8) {
            // Timber.tag("YoloDetector").d("out[0] detect start");
            for (i in 0 until output_box) {
                for (j in 0 until numClass + 5) {
                    if (isModelQuantized) {
                        out[0][i][j] =
                            oup_scale * ((byteBuffer.get().toInt() and 0xFF) - oup_zero_point)
                    } else {
                        out[0][i][j] = byteBuffer.float
                    }
                }
                // Denormalize xywh
                for (j in 0..3) {
                    out[0][i][j] *= inputSize.toFloat()
                }
            }
        } else {
            // switch the way we span through the bytebuffer
            // Timber.tag("YoloDetector V8").d("out[0] detect start");
            for (j in 0 until numClass + 4) {
                for (i in 0 until output_box) {
                    if (isModelQuantized) {
                        out[0][i][j] =
                            oup_scale * ((byteBuffer.get().toInt() and 0xFF) - oup_zero_point)
                    } else {
                        out[0][i][j] = byteBuffer.float
                    }
                }
            }
            // not always needed to denormalize for yolov8
            for (i in 0 until output_box) {
                for (j in 0..3) {
                    out[0][i][j] *= scalingFactor
                }
            }
        }
        for (i in 0 until output_box) {
            var confidence = 1.0f
            if (!isV8) {
                // confidence only valid for yolov5
                confidence = out[0][i][4]
            }
            var detectedClass = -1
            var maxClass = 0f
            val classes = FloatArray(labels.size)
            for (c in labels.indices) {
                if (!isV8) {
                    classes[c] = out[0][i][5 + c]
                } else {
                    classes[c] = out[0][i][4 + c]
                }
            }
            for (c in labels.indices) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
            val confidenceInClass = maxClass * confidence
            if (confidenceInClass > confThreshold) {
                val xPos = out[0][i][0]
                val yPos = out[0][i][1]
                val w = out[0][i][2]
                val h = out[0][i][3]
                // Timber.tag("YoloDetector").i(Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);
                val rect = RectF(
                    max(0f, xPos - w / 2),
                    max(0f, yPos - h / 2),
                    min((bitmap!!.width - 1).toFloat(), xPos + w / 2),
                    min((bitmap.height - 1).toFloat(), yPos + h / 2)
                )
                detections.add(
                    Recognition(labels[detectedClass],
                        confidenceInClass, rect, detectedClass
                    )
                )
            }
        }
        // Timber.tag("YoloDetector").d("detect end");
        return nms(detections)
    }

    fun checkInvalidateBox(
        x: Float,
        y: Float,
        width: Float,
        height: Float,
        oriW: Float,
        oriH: Float,
        intputSize: Int
    ): Boolean {
        // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        val halfHeight = height / 2.0f
        val halfWidth = width / 2.0f
        val pred_coor = floatArrayOf(x - halfWidth, y - halfHeight, x + halfWidth, y + halfHeight)

        // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        val resize_ratioW = 1.0f * intputSize / oriW
        val resize_ratioH = 1.0f * intputSize / oriH
        val resize_ratio = if (resize_ratioW > resize_ratioH) resize_ratioH else resize_ratioW //min
        val dw = (intputSize - resize_ratio * oriW) / 2
        val dh = (intputSize - resize_ratio * oriH) / 2
        pred_coor[0] = 1.0f * (pred_coor[0] - dw) / resize_ratio
        pred_coor[2] = 1.0f * (pred_coor[2] - dw) / resize_ratio
        pred_coor[1] = 1.0f * (pred_coor[1] - dh) / resize_ratio
        pred_coor[3] = 1.0f * (pred_coor[3] - dh) / resize_ratio

        // (3) clip some boxes those are out of range
        pred_coor[0] = if (pred_coor[0] > 0) pred_coor[0] else 0.0f
        pred_coor[1] = if (pred_coor[1] > 0) pred_coor[1] else 0.0f
        pred_coor[2] = if (pred_coor[2] < oriW - 1) pred_coor[2] else oriW - 1
        pred_coor[3] = if (pred_coor[3] < oriH - 1) pred_coor[3] else oriH - 1
        if (pred_coor[0] > pred_coor[2] || pred_coor[1] > pred_coor[3]) {
            pred_coor[0] = 0f
            pred_coor[1] = 0f
            pred_coor[2] = 0f
            pred_coor[3] = 0f
        }

        // (4) discard some invalid boxes
        val temp1 = pred_coor[2] - pred_coor[0]
        val temp2 = pred_coor[3] - pred_coor[1]
        val temp = temp1 * temp2
        if (temp < 0) {
            Timber.tag("checkInvalidateBox").e("temp < 0")
            return false
        }
        if (Math.sqrt(temp.toDouble()) > Float.MAX_VALUE) {
            Timber.tag("checkInvalidateBox").e("temp max")
            return false
        }
        return true
    }

    companion object {
        /**
         * Initializes a native TensorFlow session for classifying images.
         *
         * @param assetManager  The asset manager to be used to load assets.
         * @param modelFilename The filepath of the model GraphDef protocol buffer.
         * @param labelFilename The filepath of label file for classes.
         * @param isQuantized   Boolean representing model is quantized or not
         */
        @Throws(IOException::class)
        fun create(
            assetManager: AssetManager,
            modelFilename: String?,
            labelFilename: String,
            confThreshold: Float,
            isQuantized: Boolean,
            isV8: Boolean,
            outputIsScaled: Boolean,
            inputSize: Int
        ): YoloDetector {
            val d = YoloDetector()
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
                    if(compatList.isDelegateSupportedOnThisDevice){
                        // if the device has a supported GPU, add the GPU delegate
                        val delegateOptions = compatList.bestOptionsForThisDevice
                        delegateOptions.inferencePreference = INFERENCE_PREFERENCE_SUSTAINED_SPEED
                        delegateOptions.setQuantizedModelsAllowed(false)
                        delegateOptions.isPrecisionLossAllowed = true
                        this.addDelegate(GpuDelegate(delegateOptions))
                    } else {
                        // if the GPU is not supported, run on 4 threads
                        Timber.i("## USING CPU ##")
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
            if (outputIsScaled) {
                scalingFactor = 1.0f
            } else {
                scalingFactor = inputSize.toFloat()
            }
            d.isV8 = isV8
            // Pre-allocate buffers.
            val numBytesPerChannel: Int
            numBytesPerChannel = if (isQuantized) {
                1 // Quantized
            } else {
                4 // Floating point
            }
            d.inputSize = inputSize
            d.imgData =
                ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel)
            d.imgData.order(ByteOrder.nativeOrder())
            d.intValues = IntArray(d.inputSize * d.inputSize)
            if (!isV8) {
                // yolov5 case (20² + 40² + 80²)*3 = 25200
                d.output_box = ((Math.pow(
                    (inputSize / 32).toDouble(),
                    2.0
                ) + Math.pow(
                    (inputSize / 16).toDouble(),
                    2.0
                ) + Math.pow((inputSize / 8).toDouble(), 2.0)) * 3).toInt()
            } else {
                // yolov8 case (20² + 40² + 80²) = 8400
                d.output_box = (Math.pow(
                    (inputSize / 32).toDouble(),
                    2.0
                ) + Math.pow(
                    (inputSize / 16).toDouble(),
                    2.0
                ) + Math.pow((inputSize / 8).toDouble(), 2.0)).toInt()
            }
            if (d.isModelQuantized) {
                val inpten = d.tfLite!!.getInputTensor(0)
                d.inp_scale = inpten.quantizationParams().scale
                d.inp_zero_point = inpten.quantizationParams().zeroPoint
                val oupten = d.tfLite!!.getOutputTensor(0)
                d.oup_scale = oupten.quantizationParams().scale
                d.oup_zero_point = oupten.quantizationParams().zeroPoint
            }
            val shape = d.tfLite!!.getOutputTensor(0).shape()
            // Timber.i("out shape ==== "+Arrays.toString(shape));
            var numClass = 0
            if (!isV8) {
                // yolov5 case: (1, num_anchors, num_class+5)
                numClass = shape[shape.size - 1] - 5
                d.outData =
                    ByteBuffer.allocateDirect(d.output_box * (numClass + 5) * numBytesPerChannel)
            } else {
                // yolov8 case: (1, num_class+4, num_anchors)
                numClass = shape[shape.size - 2] - 4
                d.outData =
                    ByteBuffer.allocateDirect(d.output_box * (numClass + 4) * numBytesPerChannel)
            }
            d.numClass = numClass
            d.outData.order(ByteOrder.nativeOrder())
            return d
        }

        private const val NUM_THREADS = 1
        private var scalingFactor = 1.0f
    }
}