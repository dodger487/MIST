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
import android.util.Log;

import java.util.ArrayList;

/**
 * Created by cjr on 8/28/14.
 */
public class HtcOneDetectorRunnable extends DetectorRunnable {

    // Keep around a buffer of raw data points.
    private ArrayList<float[]> mSensorData;
    private ArrayList<Long> mSensorTimes;

    private static final long NS_THROWAWAY_SIZE = 500 * 1000000; // 100 ms, 2e8 ns
    private static final long NS_WAIT_SIZE = 100 * 1000000; // 100 ms, 2e8 ns
    private static final long NS_REFRESH_TIME = 350 * 1000000; // 350 ms, 3.5e8 ns

    private long mLastFiring = 0;

    private static int mXThreshold;
    private static int mYThreshold;
    private static int mZThreshold;

    HtcOneDetectorRunnable(Context context) {
        super(context);
        mSensorData = new ArrayList<float[]>();
        mSensorTimes = new ArrayList<Long>();

        mXThreshold = -3;
        mYThreshold = 15;
        mZThreshold = 6;
    }

    HtcOneDetectorRunnable(Context context, int xThreshold, int yThreshold, int zThreshold) {
        super(context);
        mSensorData = new ArrayList<float[]>();
        mSensorTimes = new ArrayList<Long>();

        mXThreshold = xThreshold;
        mYThreshold = yThreshold;
        mZThreshold = zThreshold;
    }

    public void addData(float[] values, long time) {
        while (mSensorTimes.get(0) < time - NS_THROWAWAY_SIZE){
            mSensorData.remove(0);
            mSensorTimes.remove(0); // remove anything that happened more than NS_THROWAWAY_SIZE ago
        }

        // Evaluate the model for each new data point.
        evaluateModel(time);
    }

    private void evaluateModel(long time) {
        // Only evaluate model if we haven't fired too recently and we have enough data
        if (time - mLastFiring < NS_REFRESH_TIME || mSensorData.size() < 2 ) {
            return;
        }

        // Get the index of the baseline vector
        int baseIndex = 0;
        for (int i=1; i < mSensorTimes.size(); i++) {
            if (time - mSensorTimes.get(i) < NS_WAIT_SIZE) {
                baseIndex = i;
                break;
            }
        }
//        if (baseIndex == 0) {
//            return;
//        }

        // Get the baseline vector and current values
        float[] oldValues = mSensorData.get(baseIndex);
        float[] currentValues = mSensorData.get(mSensorData.size()-1);

        Log.i(TAG, String.format("Difference X: %4f, Y: %4f, Z: %4f",
                currentValues[0] - oldValues[0],
                currentValues[1] - oldValues[1],
                currentValues[2] - oldValues[2]));

        if (currentValues[0] - oldValues[0] < mXThreshold &&
                currentValues[1] - oldValues[1] > mYThreshold &&
                currentValues[2] - oldValues[2] > mZThreshold) {
            mLastFiring = time;
            handleButtonPressed(time);
        }

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
}
