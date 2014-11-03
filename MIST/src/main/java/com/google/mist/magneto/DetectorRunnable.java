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
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Looper;
import android.util.Log;

/**
 * Created by cjr on 8/28/14.
 */
public class DetectorRunnable implements Runnable, SensorEventListener  {

    protected SensorManager mSensorManager;
    protected Sensor mMagnetometer;

    protected String TAG = "CalibratedDetectorRunnable";

    protected MagnetPullDetector.MagnetPullListener mListener = null;

    public boolean mIsCalibrated = true;

    DetectorRunnable(Context context) {
        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
//        mMagnetometer = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
    mMagnetometer = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED);
        if (mMagnetometer.getType() == Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED) {
            mIsCalibrated = false;
        }
    }

    public void setOnPullListener(MagnetPullDetector.MagnetPullListener listener) {
        mListener = listener;
    }

    protected void handleButtonPressed(long time) {
        // Fire any listeners.
        Log.i(TAG, "Button pressed.");
        if (mListener != null) {
            mListener.onMagnetPull();
        }
    }

    @Override
    public void run() {
        Looper.prepare();

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
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }
}

