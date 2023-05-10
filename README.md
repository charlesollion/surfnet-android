## Surfnet app for Android

__ Work in progress __

Welcome to the automatic Surfnet App for detecting trash from video streams. This repo uses tflite.

Tested on CPU and GPU on two Samsung S7 and S9.


### Setup

You need Android Studio with API level above 31. 

Download the tflite model and place it in app/src/main/assets/

Instructions on how to create this model to come.

### TODO

[x] Test CPU and GPU on lower end samsung phones
[ ] Monitor performances without tracking
[ ] Optimize model, delegate options and numThreads
[ ] Update old dependencies
[ ] Clean up remaining code
[ ] 