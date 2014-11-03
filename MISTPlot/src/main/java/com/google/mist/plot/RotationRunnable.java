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

package com.google.mist.plot;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Looper;
import android.util.Log;

import java.util.ArrayList;

/**
 * Created by cjr on 7/2/14.
 */
public class RotationRunnable implements Runnable, SensorEventListener {

    final String TAG = "RotationRunnable";
    int mLastAccuracy = SensorManager.SENSOR_STATUS_ACCURACY_HIGH;

    SensorManager mSensorManager;
    ArrayList<float[]> mSensorData;
    Sensor mRotationSensor;
    RotationDetector.RotationListener mListener;

    RotationRunnable(Context context) {
        mSensorData = new ArrayList<float[]>();

        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        /* From the Android documentation for Sensor.TYPE_GAME_ROTATION_VECTOR gives rotation:
         * Use this sensor in a game if you do not care about where north is, and the normal
         * rotation vector does not fit your needs because of its reliance on the magnetic field.
         */
        mRotationSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
    }

    @Override
    public void run() {
        Looper.prepare();

        mSensorManager.registerListener(this, mRotationSensor,
                SensorManager.SENSOR_DELAY_FASTEST);

        Looper.loop();
    }

    public void stop() {
        mSensorManager.unregisterListener(this);
    }

    public void setRotationListener(RotationDetector.RotationListener listener) {
        mListener = listener;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        float[] values = event.values.clone();
        Log.i(TAG, String.format("X: %4f, Y: %4f, Z: %4f, W: %2f",values[0], values[1], values[2], values[3]));
        if (mListener != null) {
            mListener.onRotationDebug(values);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        if (mLastAccuracy == SensorManager.SENSOR_STATUS_UNRELIABLE) {
            Log.i(TAG, "Calibration event?");
        }

        switch (accuracy) {
            case SensorManager.SENSOR_STATUS_ACCURACY_HIGH:
                Log.i(TAG, "Accuracy is now HIGH");
                break;
            case SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM:
                Log.i(TAG, "Accuracy is now MEDIUM");
                break;
            case SensorManager.SENSOR_STATUS_ACCURACY_LOW:
                Log.i(TAG, "Accuracy is now LOW");
                break;
            case SensorManager.SENSOR_STATUS_UNRELIABLE:
                Log.i(TAG, "Accuracy is now UNRELIABLE");
                break;
        }

        mLastAccuracy = accuracy;
    }
}
