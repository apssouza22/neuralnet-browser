import * as tf from '@tensorflow/tfjs';
import {IMAGE_H, IMAGE_W} from "./data";
import {Activation, argMax, Layer, TrainableNeuralNetwork} from "./network";

const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;

function getAvg(losses) {
    return losses
            .map((loss) => Math.abs(loss.data[0][0]))
            .reduce((a, b) => a + b) / losses.length;
}

/**
 * Get my wrapped model.
 * @param modelType
 * @returns {ModelWrapper}
 */
export function getModel(modelType) {
    let model = createModel(modelType)
    if (model instanceof TrainableNeuralNetwork) {
        return new ModelWrapper(model);
    }
    // Now that we've defined our model, we will define our optimizer. The
    // optimizer will be used to optimize our model's weight values during
    // training so that we can decrease our training loss and increase our
    // classification accuracy.

    // We are using rmsprop as our optimizer.
    // An optimizer is an iterative method for minimizing an loss function.
    // It tries to find the minimum of our loss function with respect to the
    // model's weight parameters.
    const optimizer = 'rmsprop';

    // We compile our model by specifying an optimizer, a loss function, and a
    // list of metrics that we will use for model evaluation. Here we're using a
    // categorical crossentropy loss, the standard choice for a multi-class
    // classification problem like MNIST digits.
    // The categorical crossentropy loss is differentiable and hence makes
    // model training possible. But it is not amenable to easy interpretation
    // by a human. This is why we include a "metric", namely accuracy, which is
    // simply a measure of how many of the examples are classified correctly.
    // This metric is not differentiable and hence cannot be used as the loss
    // function of the model.
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return new ModelWrapper(model);
}


function createModel(modelType) {
    if (modelType === 'ConvNet') {
        return createConvModel();
    }
    if (modelType === 'DenseNet') {
        return createDenseModel();
    }

    if (modelType === 'MyNet') {
        return createMyModel();
    }
    throw new Error(`Invalid model type: ${modelType}`);
}

export class ModelWrapper {
    /**
     * @param {tf.Sequential| TrainableNeuralNetwork} tfModel
     */
    constructor(tfModel) {
        this.model = tfModel;
    }

    /**
     * Run the model training
     * @param {int[[]]}xs
     * @param {int[[]]}ys
     * @param {{
     *     epochs: number,
     *     batchSize: number,
     *     validationSplit: number,
     *     callbacks: {
     *       onBatchEnd: (batch: number, logs: any) => void,
     *       onEpochEnd: (epoch: number, logs: any) => void
     *     }
     * }}args
     * @returns {Promise<unknown>}
     */
    async train(xs, ys, args) {
        if (this.model instanceof tf.Sequential) {
            const inputs = tf.tensor4d(xs, [xs.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
            const labels = tf.tensor2d(ys, [ys.length / NUM_CLASSES, NUM_CLASSES]);
            return this.model.fit(inputs, labels, args)
        }

        return this.#handleMyNNTraining(xs, ys, args);
    }

    #handleMyNNTraining(xs, ys, args) {
        let {images, labels} = convertFromFlattenArray(xs, ys);
        console.log('Training data size: ', images.length)
        console.log("Batch size:", args.batchSize)
        console.log("Epochs:", args.epochs)
        console.log("Validation split:", args.validationSplit)
        let i = 0;
        let epoch = 0;
        let batch = 0;
        const model = this.model;
        return new Promise(function (resolve, reject) {
            let step = () => {
                batch++;
                if (i >= images.length) {
                    i = 0;
                    epoch++;
                    args.callbacks.onEpochEnd(epoch, {val_acc: 1, loss: 0})
                }
                if (epoch + 1 >= args.epochs) {
                    resolve();
                    return;
                }
                let losses = [];
                for (let j = 0; j < args.batchSize; j++) {
                    if (i >= images.length) {
                        console.log("End of epoch " + i)
                        break;
                    }
                    losses.push(model.fit(images[i], labels[i]))
                    i++;
                }
                args.callbacks.onBatchEnd(i, {loss: getAvg(losses) * 100, val_acc: 1})
                requestAnimationFrame(step)
            };
            // using requestAnimationFrame instead of for looping to avoid blocking the UI thread
            requestAnimationFrame(step)
        });
    }

    /**
     * Evaluate the model with the given test data
     * @param {int[]} inputs
     * @param {int[]} targets
     * @returns {number} average accuracy
     */
    evaluate(xs, ys) {
        if (this.model instanceof tf.Sequential) {
            let inputs = tf.tensor4d(xs, [xs.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
            let labels = tf.tensor2d(ys, [ys.length / NUM_CLASSES, NUM_CLASSES]);
            let testResult = this.model.evaluate(inputs, labels)
            return testResult[1].dataSync()[0] * 100
        }
        let {images, labels} = convertFromFlattenArray(xs, ys);
        return this.model.evaluate(images, labels) * 100
    }

    /**
     * Perform the prediction
     * @param {int[]} input - Array of input values
     **/
    predict(xs) {
        if (this.model instanceof tf.Sequential) {
            let inputs = tf.tensor4d(xs, [xs.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
            let output = this.model.predict(inputs)

            // tf.argMax() returns the indices of the maximum values in the tensor along
            // a specific axis. Categorical classification tasks like this one often
            // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
            // one element for each output class. All values in the vector are 0
            // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
            // output from model.predict() will be a probability distribution, so we use
            // argMax to get the index of the vector element that has the highest
            // probability. This is our prediction.
            // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
            // dataSync() synchronously downloads the tf.tensor values from the GPU so
            // that we can use them in our normal CPU JavaScript code
            // (for a non-blocking version of this function, use data()).
            return Array.from(output.argMax(1).dataSync())
        }
        let {images} = convertFromFlattenArray(xs, null);
        let predictions = []
        for (const image of images) {
            let output = this.model.predict(image);
            predictions.push(argMax(output))
        }
        return predictions
    }

    summary() {
        this.model.summary()
    }
}


/**
 * Create a vanilla neural network model (My own NN)
 * @returns {TrainableNeuralNetwork}
 */
function createMyModel() {
    let imageSize = IMAGE_H * IMAGE_W;
    /**
     * @type {Layer[]} layers
     */
    let layers = []
    layers.push(new Layer(
            imageSize,
            imageSize,
            Activation.ReLU,
            Layer.INPUT
    ))
    layers.push(new Layer(
            imageSize,
            42,
            Activation.ReLU,
            Layer.HIDDEN
    ))
    layers.push(new Layer(
            42,
            10,
            Activation.SOFTMAX,
            Layer.OUTPUT
    ))
    return new TrainableNeuralNetwork(layers);
}


/**
 * Creates a model consisting of only flatten, dense and dropout layers.
 *
 * The model create here has approximately the same number of parameters
 * (~31k) as the convnet created by `createConvModel()`, but is
 * expected to show a significantly worse accuracy after training, due to the
 * fact that it doesn't utilize the spatial information as the convnet does.
 *
 * This is for comparison with the convolutional network above.
 *
 * @returns {tf.Sequential} An instance of tf.Model.
 */
function createDenseModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
    model.add(tf.layers.dense({units: 42, activation: 'relu'}));
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    return model;
}

/**
 * Creates a convolutional neural network (Convnet) for the MNIST data.
 *
 * @returns {tf.Sequential} An instance of tf.Model.
 */
function createConvModel() {
    // Create a sequential neural network model. tf.sequential provides an API
    // for creating "stacked" models where the output from one layer is used as
    // the input to the next layer.
    const model = tf.sequential();

    // The first layer of the convolutional neural network plays a dual role:
    // it is both the input layer of the neural network and a layer that performs
    // the first convolution operation on the input. It receives the 28x28 pixels
    // black and white images. This input layer uses 16 filters with a kernel size
    // of 5 pixels each. It uses a simple RELU activation function which pretty
    // much just looks like this: __/
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1],
        kernelSize: 3,
        filters: 16,
        activation: 'relu'
    }));

    // After the first layer we include a MaxPooling layer. This acts as a sort of
    // downsampling using max values in a region instead of averaging.
    // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

    // Our third layer is another convolution, this time with 32 filters.
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

    // Max pooling again.
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

    // Add another conv2d layer.
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten({}));

    model.add(tf.layers.dense({units: 64, activation: 'relu'}));

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
    // represent numbers, but it's the same idea if you had classes that
    // represented other entities like dogs and cats (two output classes: 0, 1).
    // We use the softmax function as the activation for the output layer as it
    // creates a probability distribution over our 10 classes so their output
    // values sum to 1.
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

    return model;
}



/**
 * The MNIST data is stored all images as a single array of pixel values[0,255, ...].
 * This function reshapes these into a 784 pixel values (28 * 28)
 * @param xs inputs
 * @param ys labels
 * @returns {{images: *[], labels: *[]}}
 */
export function convertFromFlattenArray(xs, ys) {
    const batchSize = xs.length / IMAGE_SIZE;
    let images = [];
    let labels = [];
    for (let i = 0; i < batchSize; i++) {
        const image = xs.slice(i * IMAGE_SIZE, (i + 1) * IMAGE_SIZE);
        images.push(image);

        if (ys) {
            const label = ys.slice(i * NUM_CLASSES, (i + 1) * NUM_CLASSES);
            labels.push(label)
        }
    }
    return {images, labels};
}
