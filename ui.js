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

const statusElement = document.getElementById('status');
const imagesElement = document.getElementById('images');

export function logStatus(message) {
    statusElement.innerText = message;
}


export function showTestResults(images, predictions, labels) {
    imagesElement.innerHTML = '';

    for (let i = 0; i < images.length; i++) {
        const image = images[i];

        const div = document.createElement('div');
        div.className = 'pred-container';

        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image, canvas);

        const pred = document.createElement('div');

        const prediction = predictions[i];
        const label = labels[i];
        const correct = prediction === label;

        pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
        pred.innerText = `pred: ${prediction}`;

        div.appendChild(pred);
        div.appendChild(canvas);

        imagesElement.appendChild(div);
    }
}

export function draw(data, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = data[i] * 255;
        imageData.data[j + 1] = data[i] * 255;
        imageData.data[j + 2] = data[i] * 255;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

export function getModelTypeId() {
    return document.getElementById('model-type').value;
}

export function getTrainEpochs() {
    return Number.parseInt(document.getElementById('train-epochs').value);
}

export function setTrainButtonCallback(callback) {
    const trainButton = document.getElementById('train');
    const modelType = document.getElementById('model-type');
    trainButton.addEventListener('click', () => {
        trainButton.setAttribute('disabled', true);
        modelType.setAttribute('disabled', true);
        callback();
    });
}
