/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as nn from './model';


import {MnistData} from './data';
import * as ui from './ui';
import {argMax} from "./network";

/**
 * Compile and train the given model.
 *
 * @param {nn.MyModel} model The model to train.
 * @param {onIterationCallback} onIteration A callback to execute every 10
 *     batches & epoch end.
 */
async function train(model, onIteration) {
  ui.logStatus('Training model...');

  // Batch size is another important hyperparameter. It defines the number of
  // examples we group together, or batch, between updates to the model's
  // weights during training. A value that is too low will update weights using
  // too few examples and will not generalize well. Larger batch sizes require
  // more memory resources and aren't guaranteed to perform better.
  const batchSize = 320;

  // Leave out the last 15% of the training data for validation, to monitor
  // overfitting during training.
  const validationSplit = 0.15;

  // Get number of training epochs from the UI.
  const trainEpochs = ui.getTrainEpochs();

  // We'll keep a buffer of loss and accuracy values over time.
  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData(null);

  const totalNumBatches = Math.ceil(trainData.xs.length / 784 * (1 - validationSplit) / batchSize) * trainEpochs;

  // During the long-running fit() call for model training, we include
  // callbacks, so that we can plot the loss and accuracy values in the page
  // as the training progresses.
  let valAcc = 1;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: onBatchEnd,
      onEpochEnd: onEpochEnd,
    }
  });

  async function onEpochEnd(epoch, logs) {
    console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss.toFixed(5));
    valAcc = logs.val_acc;
    if (onIteration) {
      onIteration('onEpochEnd', epoch, logs);
    }
      await tf.nextFrame();
  }

  async function onBatchEnd(batch, logs)  {
    trainBatchCount++;
    console.log('onBatchEnd', trainBatchCount, totalNumBatches);
    ui.logStatus(
            `Training... (` +
            `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
            ` complete). To stop training, refresh or close page.`
    );
    if (onIteration && batch % 10 === 0) {
      onIteration('onBatchEnd', batch, logs);
    }
    await tf.nextFrame();
  }

  const testAccPercent = model.evaluate(testData.xs, testData.labels);
  ui.logStatus(
      `Final test accuracy: ${testAccPercent.toFixed(1)}%`
  );
}

/**
 * Show predictions on a number of test examples.
 *
 * @param {nn.MyModel} model The model to be used for making the predictions.
 */
async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
  tf.tidy(() => {
    const predictions = model.predict(examples.xs);
    let converted = nn.convertFromFlattenArray(examples.xs, examples.labels);
    const labels = convertFromHotEncoding(converted.labels);
    ui.showTestResults(converted.images, predictions, labels);
  });
}

// converting from hot encoding to number
// [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> 3
function convertFromHotEncoding(labels) {
  let convLabels = [];
  for (const label of labels) {
    convLabels.push(argMax(label));
  }
  return convLabels;
}


let data;
async function load() {
  data = new MnistData();
  await data.load();
}

// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
ui.setTrainButtonCallback(async () => {
  ui.logStatus('Loading MNIST data...');
  await load();

  ui.logStatus('Creating model...');
  const model = nn.getModel(ui.getModelTypeId());
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model, () => showPredictions(model));
});
