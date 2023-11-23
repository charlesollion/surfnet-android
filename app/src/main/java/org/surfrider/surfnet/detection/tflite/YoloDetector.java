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

package org.surfrider.surfnet.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.surfrider.surfnet.detection.env.Utils;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import timber.log.Timber;


/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">https://github.com/tensorflow/models/tree/master/research/object_detection</a>
 * where you can find the training code.
 * <p>
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md</a>
 * - <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android">https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android</a>
 */
public class YoloDetector implements Detector {

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param isQuantized   Boolean representing model is quantized or not
     */
    public static YoloDetector create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final float confThreshold,
            final boolean isQuantized,
            final boolean isV8,
            final int inputSize)
            throws IOException {
        final YoloDetector d = new YoloDetector();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            Timber.w(line);
            d.labels.add(line);
        }
        br.close();

        try {
            Interpreter.Options options = (new Interpreter.Options());

            options.setNumThreads(NUM_THREADS);
            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.confThreshold = confThreshold;
        d.isModelQuantized = isQuantized;
        d.isV8 = isV8;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.INPUT_SIZE = inputSize;
        d.imgData = ByteBuffer.allocateDirect(d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());

        double calculation = Math.pow((inputSize / 32d), 2) + Math.pow((inputSize / 16d), 2) + Math.pow((inputSize / 8d), 2);
        if(!isV8) {
            // yolov5 case (20² + 40² + 80²)*3 = 25200
            d.output_box = (int) (calculation * 3);
        } else {
            // yolov8 case (20² + 40² + 80²) = 8400
            d.output_box = (int) calculation;
        }
        if (d.isModelQuantized){
            Tensor oupten = d.tfLite.getOutputTensor(0);
            d.oup_scale = oupten.quantizationParams().getScale();
            d.oup_zero_point = oupten.quantizationParams().getZeroPoint();
        }

        int[] shape = d.tfLite.getOutputTensor(0).shape();
        Timber.i("out shape ==== %s", Arrays.toString(shape));
        int numClass;
        if(!isV8) {
            // yolov5 case: (1, num_anchors, num_class+5)
            numClass = shape[shape.length - 1] - 5;
            d.outData = ByteBuffer.allocateDirect(d.output_box * (numClass + 5) * numBytesPerChannel);
        } else {
            // yolov8 case: (1, num_class+4, num_anchors)
            numClass = shape[shape.length - 2] - 4;
            d.outData = ByteBuffer.allocateDirect(d.output_box * (numClass + 4) * numBytesPerChannel);
        }
        d.numClass = numClass;

        d.outData.order(ByteOrder.nativeOrder());
        return d;
    }

    public int getInputSize() {
        return INPUT_SIZE;
    }

    public void setNumThreads(int num_threads) {
    }

    private void recreateInterpreter() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = new GpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }
    }

    public void useCPU() {
        recreateInterpreter();
    }

    //config yolo
    private int INPUT_SIZE = -1;

    private  int output_box;

    // Number of threads in the java app
    private static final int NUM_THREADS = 1;

    private boolean isModelQuantized;

    private boolean isV8;

    private float confThreshold;

    /** holds a gpu delegate */
    GpuDelegate gpuDelegate = null;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    // Config values.

    // Pre-allocated buffers.
    private final Vector<String> labels = new Vector<>();
    private ByteBuffer imgData;
    private ByteBuffer outData;

    private Interpreter tfLite;
    private float oup_scale;
    private int oup_zero_point;
    private int numClass;
    private YoloDetector() {
    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<>(
                            50,
                            (lhs, rhs) -> {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = Math.max(l1, l2);
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = Math.min(r1, r2);
        return right - left;
    }
    public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
        Map<Integer, Object> outputMap = new HashMap<>();

        outData.rewind();
        outputMap.put(0, outData);
        // Timber.tag("YoloDetector").d("mObjThresh: %s", getObjThresh());

        Object[] inputArray = {imgData};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
        assert byteBuffer != null;
        byteBuffer.rewind();

        ArrayList<Recognition> detections = new ArrayList<>();

        float[][][] out = new float[1][output_box][numClass + 5];
        if(!isV8) {
            // Timber.tag("YoloDetector").d("out[0] detect start");
            for (int i = 0; i < output_box; ++i) {
                for (int j = 0; j < numClass + 5; ++j) {
                    if (isModelQuantized) {
                        out[0][i][j] = oup_scale * (((int) byteBuffer.get() & 0xFF) - oup_zero_point);
                    } else {
                        out[0][i][j] = byteBuffer.getFloat();
                    }
                }
                // Denormalize xywh
                for (int j = 0; j < 4; ++j) {
                    out[0][i][j] *= getInputSize();
                }
            }
        } else {
            // switch the way we span through the bytebuffer
            // Timber.tag("YoloDetector V8").d("out[0] detect start");
            for (int j = 0; j < numClass + 4; ++j) {
                for (int i = 0; i < output_box; ++i) {
                    if (isModelQuantized) {
                        out[0][i][j] = oup_scale * (((int) byteBuffer.get() & 0xFF) - oup_zero_point);
                    } else {
                        out[0][i][j] = byteBuffer.getFloat();
                    }
                }
            }
            // no need to denormalize for yolov8
        }
        for (int i = 0; i < output_box; ++i) {
            final int offset = 0;
            float confidence = 1.0f;
            if(!isV8) {
                // confidence only valid for yolov5
                confidence = out[0][i][4];
            }
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); ++c) {
                if(!isV8) {
                    classes[c] = out[0][i][5 + c];
                } else {
                    classes[c] = out[0][i][4 + c];
                }
            }

            for (int c = 0; c < labels.size(); ++c) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > confThreshold) {
                final float xPos = out[0][i][0];
                final float yPos = out[0][i][1];

                final float w = out[0][i][2];
                final float h = out[0][i][3];
                // Timber.tag("YoloDetector").d(Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
            }
        }
        // Timber.tag("YoloDetector").d("detect end");

        return nms(detections);
    }
}
