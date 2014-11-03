/*
 * Copyright 2014 Google Inc. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mist.magneto;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.os.Looper;
import android.util.Log;

import com.google.mist.magneto.CalibratedDetectorRunnable;
import com.google.mist.magneto.MagnetPullDetector;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Detect button presses based on the magnetic input for the 3DVR box.
 * This is a classifier implemented on top of a stream of calibrated
 * magnetometer readings.
 *
 * Idea is to keep around a ring buffer of raw values representing
 * a sliding window, and evaluate metrics on this sliding window.
 *
 * We use the first value S0 (in R3) as the baseline.
 * Then we split our window in 3, and calculate metrics for each segment:
 * - offsets: |sample_i - S0|
 * - Mean (U): mean(offsets)
 * - Max (M): max(offsets)
 *
 * Next we apply thresholds on the U1, M2 and U3.
 *
 *
 * Created by cjr on 8/28/14.
 */


public class ExampleCalibratedDetectorRunnable extends DetectorRunnable {
//    private SensorManager mSensorManager;
//    private Sensor mMagnetometer;

    private String TAG = "CalibratedDetectorRunnable";

    long nsSegmentSize = 200 * 1000000;
    long nsWindowSize = nsSegmentSize * 2;

    long lastFiring = 0;

    int T1 = 30;
    int T2 = 130;

    // Keep around a buffer of raw data points.
    private ArrayList<float[]> mSensorData;
    private ArrayList<Long> mSensorTimes;

    public boolean mIsCalibrated = true;

    ExampleCalibratedDetectorRunnable(Context context) {
        super(context);
        mSensorData = new ArrayList<float[]>();
        mSensorTimes = new ArrayList<Long>();
    }

    public void addData(float[] values, long time) {
        while (mSensorTimes.get(0) < time - nsWindowSize){
            mSensorData.remove(0);
            mSensorTimes.remove(0); // remove anything that happened more than nsWindow
        }
        if (time - lastFiring < nsSegmentSize ){
            return;
        }

        // Evaluate the model for each new data point.
        evaluateModel(time);
    }

    private void evaluateModel(long time) {
        // Only evaluate model if we have enough data.
        if (mSensorData.size() < 4) {
            return;
        }

        // Calculate the baseline vector.
        float[] baseline = mSensorData.get(mSensorData.size() - 1);

        // Get the index of the sensorEvent that's under 200ms in the past
        int startSecondSegment = 0;
        for (int i=0; i < mSensorTimes.size(); i++) {
            if (time - mSensorTimes.get(i) < nsSegmentSize) {
                startSecondSegment = i;
                break;
            }
        }
        if (startSecondSegment == 0) {
            return;
        }

        float offsets[] = new float[mSensorData.size()];
        computeOffsets(offsets, baseline);
        float min1 = computeMinimum(Arrays.copyOfRange(offsets, 0, startSecondSegment));
        float max2 = computeMaximum(Arrays.copyOfRange(offsets, startSecondSegment, mSensorData.size()));

//    Log.i(TAG, String.format("Evaluated model. U1=%f, M2=%f, U3=%f", U1, M2, U3));
        if (min1 < T1 && max2 > T2) {
            handleButtonPressed(time);
        }
    }

//    private void handleButtonPressed(long time) {
//        // Fire any listeners.
//        Log.i(TAG, "Button pressed.");
//        if (mListener != null) {
//            mListener.onMagnetPull();
//        }
//    }

    private void computeOffsets(float[] offsets, float[] baseline) {
        // TODO(cjr): the "output" of this is given as offsets, an input... is that okay style?
        for (int i = 0; i < mSensorData.size(); i++) {
            float[] point = mSensorData.get(i);
            float[] o = {point[0] - baseline[0], point[1] - baseline[1], point[2] - baseline[2]};
            float magnitude = (float) Math.sqrt(o[0]*o[0] + o[1]*o[1] + o[2]*o[2]);
            offsets[i] = magnitude;
        }
    }

    private float computeMean(float[] offsets) {
        float sum = 0;
        for (float o : offsets) {
            sum += o;
        }
        return sum/offsets.length;
    }

    private float computeMaximum(float[] offsets) {
        float max = Float.NEGATIVE_INFINITY;
        for (float o : offsets) {
            if (o > max) {
                max = o;
            }
        }
        return max;
    }

    private float computeMinimum(float[] offsets) {
        float min = Float.POSITIVE_INFINITY;
        for (float o : offsets) {
            if (o < min) {
                min = o;
            }
        }
        return min;
    }

    @Override
    public void run() {
        Looper.prepare();
//      Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
//      Process.setThreadPriority(Process.THREAD_PRIORITY_LOWEST);

        mSensorManager.registerListener(this, mMagnetometer,
                SensorManager.SENSOR_DELAY_FASTEST);

        Looper.loop();
    }

    public void stop() {
        mSensorManager.unregisterListener(this);
    }


    /**
     * @return false only if failing to switch to uncalibrated magnetometer.
     */
    public boolean toggleCalibrated() {
        switch (mMagnetometer.getType()) {
            case Sensor.TYPE_MAGNETIC_FIELD:
                Sensor newSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED);
                if (newSensor == null) {
                    return false;
                } else {
                    mIsCalibrated = false;
                }
                mMagnetometer = newSensor;
                break;
            case Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED:
                mMagnetometer = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
                mIsCalibrated = true;
                break;
            default:
                Log.e(TAG, "Unknown sensor type.");
        }
        return true;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.equals(mMagnetometer)) {
            // Discard all empty vectors. This often happens on Nexus 4.
            if (event.values[0] == 0 && event.values[1] == 0 && event.values[2] == 0) {
                return;
            }

            float[] values = {event.values[0], event.values[1], event.values[2]};
            mSensorData.add(values);
            mSensorTimes.add(event.timestamp);
            addData(values, event.timestamp);

            float magnitude = (float)Math.sqrt(values[0] * values[0]
                    + values[1] * values[1] + values[2] * values[2]);
            Log.i(TAG, String.format("Accuracy: %d, X: %4f, Y: %4f, Z: %4f, Mag: %4f",
                    event.accuracy, values[0], values[1], values[2], magnitude));
            if (mMagnetometer.getType() ==  Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED) {
        /* Values 3-5 are the current estimated "noise" for those axis. Subtract them to get
           the uncalibrated version
        */
                Log.i(TAG, String.format("BIAS X: %4f, Y: %4f, Z: %4f", event.values[3], event.values[4],
                        event.values[5]));
            }

            if (mListener != null) {
                mListener.onMagnetDataDebug(event);
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        switch (accuracy) {
            case SensorManager.SENSOR_STATUS_ACCURACY_HIGH:
                Log.i(TAG, "Accuracy changed is now HIGH");
                break;
            case SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM:
                Log.i(TAG, "Accuracy changed is now MEDIUM");
                break;
            case SensorManager.SENSOR_STATUS_ACCURACY_LOW:
                Log.i(TAG, "Accuracy changed is now LOW");
                break;
            case SensorManager.SENSOR_STATUS_UNRELIABLE:
                Log.i(TAG, "Accuracy changed is now UNRELIABLE");
                break;
            default:
                Log.i(TAG, String.format("Accuracy changed: unexpected accuracy %d", accuracy));
        }

        if (mListener != null) {
            mListener.onMagnetAccuracyChanged(accuracy);
        }
    }
}
