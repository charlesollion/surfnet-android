## Surfnet app for Android

*Work in progress*

Welcome to the automatic Surfnet App for detecting trash from video streams. This repo uses tflite.

Tested on CPU and GPU on two Samsung S7 and S9.
Works in Android Emulator, without GPU support.


### Setup

You need Android Studio with API level above 31. 

Instructions on how to create your own model to come.

### TODO

- [x] Test CPU and GPU on lower end samsung phones
- [ ] Monitor performances without tracking
- [ ] Optimize model, delegate options and numThreads
- [ ] Update old dependencies
- [ ] Clean up remaining code
- [ ] Check performances in landscape mode
- [ ] Emulator with local Video (and GPU ?)