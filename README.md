## Surfnet app for Android

*Work in progress*

Welcome to the automatic Surfnet App for detecting trash from video streams. This repo uses tflite.

Tested on CPU and GPU on two Samsung S7 and S9.
Works in Android Emulator, without GPU support.


### Setup

You need Android Studio with API level above 31. 
You need to set up OpenCV, here are instructions on Android Studio:
- Download opencv sdk (tested with [4.8.0](https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-android-sdk.zip))
- File > New > Import Module, select the sdk folder, name it `:opencv`
- Change opencv's `build.gradle` to match android target and min sdk version of the main app
- File > Project Structure > Add dependency (module) > select 'app' > click `opencv`