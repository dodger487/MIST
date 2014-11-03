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
import android.graphics.*;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import java.util.Date;

import java.util.ArrayList;

/**
 * Plots a graph of data in split screen view.
 * Created by smus on 4/30/14.
 */
public class VRPlotView extends View {

  String TAG = "VRPlotView";
  ArrayList<float[]> mAxisData;
  Paint COLORS[] = {new Paint(), new Paint(), new Paint(), new Paint()};
  Paint mDividerPaint = new Paint();
  Paint mLabelPaint = new Paint();

  int PADDING_X = 100;
  int PADDING_Y = 200;

  // Number of samples (depending on the size of the canvas).
  int mMaxSamples = 0;

  // Temporary bitmap and canvas.
  Bitmap mBitmap;
  Canvas mCanvas;

  // Min and max values of the plot.
  float mMin = 0;
  float mMax = 0;

  // Magneto.
  private Bitmap mMagnetoIcon;

  // An optional label to persistently show front-and-center.
  private String mLabel;

  // Whether or not to flash the screen.
  private Date mFlashStart = null;

  int mIteration = 0;


  public VRPlotView(Context context, AttributeSet attrs) {
    super(context, attrs);

    mAxisData = new ArrayList<float[]>();
    COLORS[0].setColor(Color.RED);
    COLORS[1].setColor(Color.GREEN);
    COLORS[2].setColor(Color.BLUE);
    COLORS[3].setColor(Color.BLACK);

    mDividerPaint.setColor(Color.DKGRAY);
    mDividerPaint.setStrokeWidth(2);

    mLabelPaint.setColor(Color.LTGRAY);
    mLabelPaint.setStrokeWidth(2);
    mLabelPaint.setTextSize(34);
    mLabelPaint.setTextAlign(Paint.Align.CENTER);


    for (Paint p : COLORS) {
      p.setStrokeWidth(3);
    }

    mMagnetoIcon = BitmapFactory.decodeResource(getResources(), R.drawable.magneto);
  }


  public void flashScreen() {
    mFlashStart = new Date();
  }

  public void showLabel(String label) {
    mLabel = label;
  }


  @Override
  protected void onSizeChanged(int w, int h, int oldw, int oldh){
    super.onSizeChanged(w, h, oldw, oldh);
    Log.i(TAG, String.format("Size changed to %d x %d px.", w, h));

    mBitmap = Bitmap.createBitmap(w/2, h, Bitmap.Config.ARGB_8888);
    mCanvas = new Canvas(mBitmap);

    mMaxSamples = mBitmap.getWidth() - PADDING_X *2;
  }


  @Override
  protected void onDraw(Canvas canvas) {
    // Clear the temporary canvas.
    mCanvas.drawColor(Color.WHITE);

    if (mIteration % 10 == 0) {
      reCalculateBounds();
    }

    // Draw each line
    for (int i = 0; i < mAxisData.size() - 1; i++) {
      float[] point = mAxisData.get(i);
      float[] nextPoint = mAxisData.get(i+1);
      for (int j = 0; j < point.length; j++) {
        drawLine(point, nextPoint, j, i);
      }
    }
    // Draw magneto in the lower middle part.
    int magnetoX = mCanvas.getWidth()/2 - mMagnetoIcon.getWidth()/2;
    mCanvas.drawBitmap(mMagnetoIcon, magnetoX, mCanvas.getHeight() - PADDING_X, null);

    // Draw the left and right canvases (they are identical).
    canvas.drawBitmap(mBitmap, 0, 0, null);
    canvas.drawBitmap(mBitmap, mCanvas.getWidth(), 0, null);

    // Draw a line for the middle divider.
    canvas.drawLine(mCanvas.getWidth()-1, 0, mCanvas.getWidth()-1, canvas.getHeight(), mDividerPaint);

    // Draw the label if there is one.
    if (mLabel != null) {
      canvas.drawText(mLabel, mCanvas.getWidth(), 60, mLabelPaint);
    }

    setBackgroundColor(Color.BLACK);
    setAlpha(0.5f);

    // Draw an overlay.
    mCanvas.drawPaint(getMagnetOverlay());

    mIteration++;
  }

  private Paint getMagnetOverlay() {
    Paint out = new Paint();
    out.setAlpha(0);
    if (mFlashStart == null) {
      return out;
    }
    int maxAlpha = 200;
    int fadeInTime = 100;
    int fadeOutTime = 100;
    // How far we are into the animation
    long duration = new Date().getTime() - mFlashStart.getTime();
    int color = Color.BLACK;
    int alpha = 0;
    if (duration < fadeInTime) {
      // Fading in. Calculate the alpha.
      float percent = (float)duration / (float)fadeInTime;
      alpha = (int) (maxAlpha * percent);
    } else if (duration < fadeOutTime) {
      // Fading out. Calculate the alpha.
      float percent = (float)(duration - fadeInTime) / (float)fadeOutTime;
      alpha = (int) (maxAlpha * (1 - percent));
    }
    Log.d(TAG, String.format("Alpha: %d", alpha));
    out.setColor(color);
    out.setAlpha(alpha);
    return out;
  }

  private void reCalculateBounds() {
    mMin = Float.POSITIVE_INFINITY;
    mMax = Float.NEGATIVE_INFINITY;

    // Go through all data points computing min and max.
    for (float[] point : mAxisData) {
      for (float datum : point) {
        if (datum < mMin) {
          mMin = datum;
        }
        if (datum > mMax) {
          mMax = datum;
        }
      }
    }
  }

  private void drawLine(float[] point, float[] nextPoint, int axis, int time) {
    float range = mMax - mMin;

    // Calculate the percentages of the available space to render.
    float p1 = ((point[axis] - mMin) / range);
    float p2 = ((nextPoint[axis] - mMin) / range);

    // Convert percent into coordinates.
    float y1 = PADDING_Y + p1 * (mCanvas.getHeight() - PADDING_Y *2);
    float y2 = PADDING_Y + p2 * (mCanvas.getHeight() - PADDING_Y *2);
    float x = time + PADDING_X;

    // Draw the line.
    mCanvas.drawLine(x, y1, x + 1, y2, getPaint(axis));
  }

  public void addData(float[] data) {
    mAxisData.add(data);
    if (mAxisData.size() >= mMaxSamples) {
      mAxisData.remove(0);
    }
    postInvalidate();
  }

  private Paint getPaint(int index) {
    if (index < COLORS.length) {
      return COLORS[index];
    }
    return null;
  }
}
