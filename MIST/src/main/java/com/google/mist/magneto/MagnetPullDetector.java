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
import android.hardware.SensorEvent;
import android.os.Build;

/**
 * Interface to the magnet input detector. This class creates a separate
 * input thread for subscribing to the magnetometer, lets clients subscribe to the magnetic pull event.
 *
 * Created by smus on 5/7/14.
 */
public class MagnetPullDetector {

//    private CalibratedDetectorRunnable mDetector;
  private DetectorRunnable mDetector;
  private Thread mDetectorThread;

  public MagnetPullDetector(Context context) {
    if (Build.MODEL.equals("HTC One")) {
        mDetector = new HtcOneDetectorRunnable(context);
    } else {
        mDetector = new ExampleCalibratedDetectorRunnable(context);
    }
  }

  public void start() {
    mDetectorThread = new Thread(mDetector);
    mDetectorThread.setPriority(Thread.MAX_PRIORITY);
    mDetectorThread.start();
  }

  public void stop() {
    if (mDetectorThread != null) {
      mDetectorThread.interrupt();
      mDetector.stop();
    }
  }

  public void setOnPullListener(MagnetPullListener listener) {
    mDetector.setOnPullListener(listener);
  }

  public boolean toggleCalibrated() {
    return mDetector.toggleCalibrated();
  }

  public boolean getCalibrationStatus() { return mDetector.mIsCalibrated; }

  public interface MagnetPullListener {
    void onMagnetPull();
    void onMagnetDataDebug(SensorEvent event);
    void onMagnetAccuracyChanged(int newAccuracy);
  }
}
