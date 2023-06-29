package org.surfrider.surfnet.detection.tflite;

import android.content.res.AssetManager;

import java.io.IOException;

public class DetectorFactory {
    public static YoloDetector getDetector(
            final AssetManager assetManager,
            final String modelFilename)
            throws IOException {
        String labelFilename = null;
        boolean isQuantized = false;
        boolean isV8 = false;
        int inputSize = 0;

        if (modelFilename.equals("yolov5sCoco.tflite")) {
            labelFilename = "file:///android_asset/coco.txt";
            inputSize = 640;
        }
        else if (modelFilename.equals("yolov5sSurf.tflite")) {
            labelFilename = "file:///android_asset/labelmap_surfnet.txt";
            inputSize = 640;
        }
        else if (modelFilename.equals("yolo_surfnet_da12-fp16.tflite")) {
            labelFilename = "file:///android_asset/labelmap_surfnet.txt";
            inputSize = 640;
        }
        else if (modelFilename.equals("yolov8n_float16.tflite")) {
            labelFilename = "file:///android_asset/coco.txt";
            inputSize = 640;
            isV8 = true;
        }

        return YoloDetector.create(assetManager, modelFilename, labelFilename, isQuantized,
                isV8, inputSize);
    }

}
