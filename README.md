## Surfnet app for Android

*Work in progress*

Welcome to the automatic Surfnet App for detecting trash from video streams. This repo uses tflite.

Tested on CPU and GPU on two Samsung S7 and S9.
Works in Android Emulator, without GPU support.


### Setup

You need Android Studio with API level above 31. 
You need to set up OpenCV, here are instructions on Android Studio:
- Delete everything related to opencv in `build.gradle` and `settings.gradle` files
- Download opencv sdk (tested with [4.8.0](https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-android-sdk.zip))
- File > New > Import Module, select the sdk folder, name it `:opencv`
- Change opencv's `build.gradle` to match android target and min sdk version of the main app
- In the same file, make sure to add this line under compileSdkVersion `namespace 'org.opencv'`
- *optional* change Java version in `compileOptions` to make sure to match the one in your app's `build.gradle`