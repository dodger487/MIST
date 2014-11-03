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

import android.app.Activity;
import android.content.Context;
import android.graphics.Point;
import android.hardware.SensorEvent;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Vibrator;
import android.util.Log;
import android.view.Display;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup.LayoutParams;
import android.view.WindowManager;
import android.widget.Toast;
import com.google.mist.magneto.MagnetPullDetector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class MainActivity extends Activity implements MagnetPullDetector.MagnetPullListener, RotationDetector.RotationListener {

    String TAG = "Magneto";

    private final int POSITIVE_LABEL = 1;

    private final int UP_LABEL = 2;
    private final int DOWN_LABEL = 3;
    private final int LEFT_LABEL = 4;
    private final int RIGHT_LABEL = 5;


    private enum RecordingType {
        POSITIVE, NEGATIVE, NONE
    }

    private ArrayList<float[]> mRotationData;
    private ArrayList<Long> mRotationTime;

    private boolean mRecordRotation;
    private RotationDetector mRotationDetector;

    private ArrayList<float[]> mSensorData; // For the magnetometer
    private ArrayList<Long> mSensorTime;
    private ArrayList<Integer> mSensorAccuracy;
    private View mDecorView;

    private ArrayList<Long> mAccuracyTime; // For onAccuracyChanged event
    private ArrayList<Integer> mAccuracyData;

    private ArrayList<Long> mPositivesTime; // For true positives of magnet pull
    private ArrayList<Integer> mPositivesData; // Labels, in case we have multiple types of events

    private VRPlotView mVRPlotView;

    private MagnetPullDetector mPullDetector;

    private boolean mIsRecording = false;
    private String mDumpPath = null;
    private RecordingType mRecordingType = RecordingType.NONE;
    private String mSessionId;

    private Vibrator mVibrator;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        mDecorView = getWindow().getDecorView();
        mVRPlotView = new VRPlotView(this, null);
        addContentView(mVRPlotView, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        mSensorData = new ArrayList<float[]>();
        mSensorTime = new ArrayList<Long>();
        mSensorAccuracy = new ArrayList<Integer>();

        mAccuracyData = new ArrayList<Integer>();
        mAccuracyTime = new ArrayList<Long>();

        mRotationData = new ArrayList<float[]>();
        mRotationTime = new ArrayList<Long>();

        mPullDetector = new MagnetPullDetector(this);
        mPullDetector.setOnPullListener(this);

        mPositivesTime = new ArrayList<Long>();
        mPositivesData = new ArrayList<Integer>();

        mVibrator = (Vibrator) getApplicationContext().getSystemService(Context.VIBRATOR_SERVICE);

        mRecordRotation = false; // You can record rotation here in addition to magnetometer readings.
        if (mRecordRotation) {
            mRotationDetector = new RotationDetector(this);
            mRotationDetector.setRotationListener(this);
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private String createSessionId() {
        Random random = new Random();

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 5; i++) {
            int rand = random.nextInt(10);
            sb.append(String.valueOf(rand));
        }
        return String.format("session-%s", sb.toString());
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {

        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            toggleRecording(RecordingType.POSITIVE);
            return true;
        }

        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            toggleRecording(RecordingType.NEGATIVE);
            return true;
        }

        /* To label magnet pulls with ground truth, connect with ADB and run command:
           $ input keyevent 66
           or from console, not inside adb shell,
           $ adb shell input keyevent 66
           or connect a bluetooth keyboard and hit "enter" key
         */
        if (mIsRecording && (keyCode == KeyEvent.KEYCODE_ENTER ||
                keyCode == KeyEvent.KEYCODE_DPAD_UP ||
                keyCode == KeyEvent.KEYCODE_DPAD_DOWN ||
                keyCode == KeyEvent.KEYCODE_DPAD_LEFT ||
                keyCode == KeyEvent.KEYCODE_DPAD_RIGHT)
                ) {
            mVibrator.vibrate(30);
            int recordedLabel = 0;
            switch (keyCode) {
                case KeyEvent.KEYCODE_ENTER:
                    recordedLabel = POSITIVE_LABEL;
                    break;
                case KeyEvent.KEYCODE_DPAD_UP:
                    recordedLabel = UP_LABEL;
                    break;
                case KeyEvent.KEYCODE_DPAD_DOWN:
                    recordedLabel = DOWN_LABEL;
                    break;
                case KeyEvent.KEYCODE_DPAD_LEFT:
                    recordedLabel = LEFT_LABEL;
                    break;
                case KeyEvent.KEYCODE_DPAD_RIGHT:
                    recordedLabel = RIGHT_LABEL;
                    break;
            }
            long lastFiring = mSensorTime.get(mSensorTime.size() - 1);
            mPositivesTime.add(lastFiring);
            mPositivesData.add(recordedLabel);
        }

        return super.onKeyDown(keyCode, event);
    }

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    /**
     *
     * @param recordingType The type of recording to make, if we're starting a new recording.
     */
    private void toggleRecording(RecordingType recordingType) {
        // Toggle between recording and not.
        if (mIsRecording) {
            stopRecording();
            toast(String.format("Stopped recording. Dumped data to %s.", mDumpPath));
        } else {
            mRecordingType = recordingType;
            startRecording();
            toast("Started recording");
        }
    }

    private void stopRecording() {
        // Dump all sensor data to the SD card.
        try {
            dumpSensorData();
        } catch (JSONException e) {
            Log.e(TAG, "Error writing JSON!");
        }
        mIsRecording = false;
        mRecordingType = RecordingType.NONE;

        // Hide the label.
        mVRPlotView.showLabel(null);

        // Clear the data.
        mSensorData.clear();
        mSensorTime.clear();
        mSensorAccuracy.clear();
        mPositivesData.clear();
        mPositivesTime.clear();
        mRotationData.clear();
        mRotationTime.clear();
        mAccuracyTime.clear();
        mAccuracyData.clear();
    }

    private void startRecording() {
        mIsRecording = true;
        String type = null;
        switch (mRecordingType) {
            case POSITIVE:
                type = "magnet clicks";
                break;
            case NEGATIVE:
                type = "motion";
                break;
            default:
                break;
        }
        mVRPlotView.showLabel(String.format("Recording %s for %s", type, mSessionId));
    }

    @Override
    public void onResume() {
        // Create unique session ID where to save all of the information about the collection.
        mSessionId = createSessionId();

        if (mRecordRotation) {
            mRotationDetector.start();
        }
        mPullDetector.start();
        super.onResume();
    }

    @Override
    public void onPause() {
        mPullDetector.stop();
        if (mRecordRotation) {
            mRotationDetector.stop();
        }
        super.onPause();
    }

    private void dumpSensorData() throws JSONException { //TODO(cjr): do we want JSONException here?
        File dataDir = getOrCreateSessionDir();
        File target = new File(dataDir, String.format("%s.json", mRecordingType));

        JSONObject jsonObject = new JSONObject();
        jsonObject.put("version", "1.0.0");

        boolean isCalibrated = mPullDetector.getCalibrationStatus();
        jsonObject.put("calibrated", isCalibrated);

        // Write system information
        Display display = getWindowManager().getDefaultDisplay();
        Point size = new Point();
        display.getSize(size);
        JSONObject deviceData = new JSONObject();
        deviceData.put("Build.DEVICE", Build.DEVICE);
        deviceData.put("Build.MODEL", Build.MODEL);
        deviceData.put("Build.PRODUCT", Build.PRODUCT);
        deviceData.put("Build.VERSION.SDK_INT", Build.VERSION.SDK_INT);
        deviceData.put("screenResolution.X", size.x);
        deviceData.put("screenResolution.Y", size.y);
        jsonObject.put("systemInfo", deviceData);

        // Write magnetometer data
        JSONArray magnetData = new JSONArray();
        for (int i = 0; i < mSensorData.size(); i++) {
            JSONArray dataPoint = new JSONArray();
            long time = mSensorTime.get(i);
            dataPoint.put(time);
            dataPoint.put(mSensorAccuracy.get(i));
            float[] data = mSensorData.get(i);
            for (float d : data) {
                dataPoint.put(d);
            }
            magnetData.put(dataPoint);
        }
        jsonObject.put("magnetometer", magnetData);

        // Write onAccuracyChanged data
        JSONArray accuracyChangedData = new JSONArray();
        for (int i = 0; i < mAccuracyData.size(); i++) {
            JSONArray dataPoint = new JSONArray();
            long time = mAccuracyTime.get(i);
            dataPoint.put(time);
            dataPoint.put(mAccuracyData.get(i));
            accuracyChangedData.put(dataPoint);
        }
        jsonObject.put("onAccuracyChangedData", accuracyChangedData);

        // Write rotation data
        if (mRecordRotation) {
            JSONArray rotationData = new JSONArray();
            for (int i = 0; i < mSensorData.size(); i++) {
                JSONArray dataPoint = new JSONArray();
                long time = mRotationTime.get(i);
                dataPoint.put(time);
                float[] data = mRotationData.get(i);
                for (float d : data) {
                    dataPoint.put(d);
                }
                rotationData.put(dataPoint);
            }
            jsonObject.put("rotation", rotationData);
        }

        // Write event labels
        JSONArray trueLabels = new JSONArray();
        for (int i = 0; i < mPositivesData.size(); i++) {
            JSONArray dataPoint = new JSONArray();
            long time = mPositivesTime.get(i);
            dataPoint.put(time);
            dataPoint.put(mPositivesData.get(i));
            trueLabels.put(dataPoint);
        }
        jsonObject.put("labels", trueLabels);

        try {
            FileWriter fw = new FileWriter(target, true);
            fw.write(jsonObject.toString());
            fw.flush();
            fw.close();
            mDumpPath = target.toString();
        } catch (IOException e) {
            Log.e(TAG, e.toString());
        }
    }

    private File getOrCreateSessionDir() {
        File rootDir = Environment.getExternalStorageDirectory();
        File dataDir = new File(rootDir, String.format("/magneto/%s/", mSessionId));

        if (!dataDir.exists()) {
            // Create it if it doesn't exist yet.
            boolean result = dataDir.mkdirs();
            Log.i(TAG, String.format("Made dir: %b", result));
        }
        return dataDir;
    }

    private void dumpSystemInfo() {
        File dataDir = getOrCreateSessionDir();
        File target = new File(dataDir, "system_info.txt");

        // Only write system info once.
        if (target.exists()) {
            return;
        }

        try {
            Display display = getWindowManager().getDefaultDisplay();
            Point size = new Point();
            display.getSize(size);

            FileWriter fw = new FileWriter(target, true);
            fw.write(String.format("Build.DEVICE: %s\n", Build.DEVICE));
            fw.write(String.format("Build.MODEL: %s\n", Build.MODEL));
            fw.write(String.format("Build.PRODUCT: %s\n", Build.PRODUCT));
            fw.write(String.format("Build.VERSION.SDK_INT: %d\n", Build.VERSION.SDK_INT));
            fw.write(String.format("Screen resolution: %d x %d px\n", size.x, size.y));
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            mDecorView.setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION // hide nav bar
                            | View.SYSTEM_UI_FLAG_FULLSCREEN // hide status bar
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            );
        }
    }

    @Override
    public void onMagnetDataDebug(SensorEvent event) {
        // The first three values are the output from magnetometer
        // If magnetometer is uncalibrated, we'll get more values, so let's only take these three
        float[] values = {event.values[0], event.values[1],event.values[2]};
        mVRPlotView.addData(values);

        if (mIsRecording) {
            mSensorTime.add(event.timestamp);
            mSensorData.add(values);
            mSensorAccuracy.add(event.accuracy);
        }
    }

    @Override
    public void onRotationDebug(float[] values) {
        if (mIsRecording) {
            // Save times as nanoseconds, because SensorEvent.timestamp is formatted that way.
            mRotationTime.add(System.nanoTime());
            mRotationData.add(values);
        }
    }

    @Override
    public void onMagnetPull() {
        mVRPlotView.flashScreen();
        Log.v(TAG, "Firing: it fired!");
//        mVibrator.vibrate(30);
    }

    @Override
    public void onMagnetAccuracyChanged(int newAccuracy) {
        if (mIsRecording) {
            // Save times as nanoseconds, because SensorEvent.timestamp is formatted that way.
            mAccuracyTime.add(System.nanoTime());
            mAccuracyData.add(newAccuracy);
        }
    }
}
