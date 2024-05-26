Decoupling test videos: This is a video recorded during a decoupling test. When the alarm sounds, it is recognized as a decouple state, and when the alarm stops, it is recognized as a couple state.

TP folder: Actual: decouple / measured: decouple

TN folder: Actual: couple / measured: couple

FP folder: Actual: coupe / measured: decouple

FN folder: Actual: decouple / measured: couple

=================================================================

Usingsensor_decoupling_detection.zip: Decoupling Test Application

The monitor and detector modes are implemented in the application

If you want to use it, you need a server that has the activity recognition and decoupling detection algorithm.

=================================================================

Test and train Data / Machine-learning models are at main branch
This test data is feature data extracted through a detector after collecting 300 pieces of data using an application dedicated to accelerometer data collection.

===================================================================

The code for server that is used to collect data and determine decoupling status is at django branch

===================================================================

In the ensemble branch, there is the test data, which is collected by using sensor application, training code for machine-learning models, and the performance evaluation code for the ensemble.
The results of activity recognition in the paper use this test data.
