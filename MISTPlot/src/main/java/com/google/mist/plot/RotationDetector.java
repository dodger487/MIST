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

/**
 * Created by cjr on 7/2/14.
 */
public class RotationDetector {

    private RotationRunnable mRotationRunnable;
    private Thread mRotationThread;

    public RotationDetector(Context context) {
        mRotationRunnable = new RotationRunnable(context);
    }

    public void start() {
        mRotationThread = new Thread(mRotationRunnable);
        mRotationThread.start();
    }

    public void stop() {
        if (mRotationThread != null) {
            mRotationThread.interrupt();
            mRotationRunnable.stop();
        }
    }

    public void setRotationListener(RotationListener listener) {
        mRotationRunnable.setRotationListener(listener);
    }

    public interface RotationListener {
        void onRotationDebug(float[] vector);
    }
}
