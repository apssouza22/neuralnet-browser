// This file contain our vanilla javascript code for the neural network
import {Matrix} from './matrix';

/**
 * Artificial Neural Network
 */
class NeuralNetwork {

    /**
     * @param  {Layer[]} layers - network layers
     */
    constructor(layers) {
        this.layerNodesCounts = []; // no of neurons per layer
        this.layers = layers;
        this.#setLayerNodeCounts(layers);
    }

    #setLayerNodeCounts(layers) {
        for (const layer of layers) {
            if (layer.layerType == Layer.INPUT) {
                continue;
            }
            this.layerNodesCounts.push(layer.weights.cols);
            if (layer.layerType == Layer.OUTPUT) {
                this.layerNodesCounts.push(layer.weights.rows);
            }
        }
    }

    /**
     * Perform the feed forward operation
     * @param {number[]} input_array - Array of input values
     * @param {Boolean} GET_ALL_LAYERS - if we need all layers after feed forward instead of just output layer
     * @returns {number[]} - the Neural net output for each layer
     */
    feedForward(input_array, GET_ALL_LAYERS = false) {
        this.#feedforwardArgsValidator(input_array)
        let inputMat = Matrix.fromArray(input_array)
        let outputs = [];
        for (let i = 0; i < this.layerNodesCounts.length; i++) {
            outputs[i] = this.layers[i].processFeedForward(inputMat);
            inputMat = outputs[i];
        }

        if (GET_ALL_LAYERS == true) {
            return outputs;
        }
        return outputs[outputs.length - 1].toArray();
    }


    // Argument validator functions
    #feedforwardArgsValidator(input_array) {
        if (input_array.length != this.layers[0].inputs.length) {
            throw new Error("Feedforward failed : Input array and input layer size doesn't match.");
        }
    }
}

/**
 * Neural Network that implements backpropagation algorithm
 */
export class TrainableNeuralNetwork extends NeuralNetwork {
    learningRate;

    /**
     * Constructor
     * @param {Layer[]}layers
     * @param [float]learningRate
     */
    constructor(layers, learningRate = 0.1) {
        super(layers);
        this.learningRate = learningRate;
    }

    /**
     * Trains with back propagation
     * @param {int[]} input - Array of input values
     * @param {int[]} target - Array of labels
     */
    fit(input, target) {
        this.#trainArgsValidator(input, target)
        let pred = this.feedForward(input, true);
        console.log("Predicted", pred)
        console.log(target, pred)
        let loss = this.calculateLoss(target);
        this.updateWeights();
        return loss;
    }

    /**
     * Evaluate the model with the given test data
     * @param {int[]} inputs
     * @param {int[]} targets
     * @returns {number} average accuracy
     */
    evaluate(inputs, targets) {
        let total = 0
        for (let i = 0; i < inputs.length; i++) {
            const output = this.predict(inputs[i]);
            total += argMax(output) == argMax(targets[i]) ? 1 : 0;
        }
        return total / inputs.length;
    }

    /**
     * Perform the prediction
     * @param {int[]} input - Array of input values
     **/
    predict(input) {
        return this.feedForward(input, false);
    }

    /**
     * Calculate the loss for each layer
     * @param {int[]} target - Array of out values
     **/
    calculateLoss(target) {
        const targetMatrix = Matrix.fromArray(target)
        this.#loopLayersInReverse(this.layerNodesCounts, (layerIndex) => {
            let prevLayer
            if (this.layers[layerIndex].layerType != Layer.OUTPUT) {
                prevLayer = this.layers[layerIndex + 1]
            }
            this.layers[layerIndex].calculateErrorLoss(targetMatrix, prevLayer);
        })
        return this.layers[this.layers.length - 1].layerError;
    }

    /**
     * Neural network summary
     */
    summary() {
        console.log("Neural Network Summary");
        console.log("Layers : ", this.layerNodesCounts);
        console.log("Learning Rate : ", this.learningRate);
    }

    /**
     * Update the weights of each layer based on the loss calculated
     */
    updateWeights() {
        this.#loopLayersInReverse(this.layerNodesCounts, (layerIndex) => {
            const currentLayer = this.layers[layerIndex]
            const nextLayer = this.layers[layerIndex - 1]
            currentLayer.calculateGradient(this.learningRate);
            currentLayer.updateWeights(nextLayer.outputs);
        })
    }

    #loopLayersInReverse(layerOutputs, callback) {
        for (let layer_index = layerOutputs.length - 1; layer_index >= 1; layer_index--) {
            callback(layer_index)
        }
    }


    #trainArgsValidator(input_array, target_array) {
        if (input_array.length != this.layerNodesCounts[0]) {
            throw new Error("Training failed : Input array and input layer size doesn't match.");
        }
        if (target_array.length != this.layerNodesCounts[this.layerNodesCounts.length - 1]) {
            throw new Error("Training failed : Target array and output layer size doesn't match.");
        }
    }

}

/**
 * Available activation functions
 */
export class Activation {
    static SIGMOID = 1;
    static ReLU = 2;
    static SOFTMAX = 3;

    /**
     * Create a new activation function pair (activation and derivative)
     * @param {int} activationType
     * @returns {{
     *   derivative: ((function(number): (number))),
     *   activation: ((function(number): (number)))
     * }}
     */
    static create(activationType) {
        switch (activationType) {
            case Activation.SIGMOID:
                return {
                    activation: Activation.#sigmoid,
                    derivative: Activation.#sigmoid_derivative
                }

            case Activation.ReLU:
                return {
                    activation: Activation.#relu,
                    derivative: Activation.#relu_derivative
                }
            case Activation.SOFTMAX:
                return {
                    activation: Activation.#softmax,
                    derivative: Activation.#softmax_derivative
                }
            default:
                console.error('Activation type invalid, setting sigmoid by default');
                return {
                    activation: Activation.sigmoid,
                    derivative: Activation.sigmoid_derivative
                }
        }
    }

    static #softmax_derivative(y) {
        return y * (1 - y);
    }

    static #softmax(x) {
        return 1 / (1 + Math.exp(-x));
    }

    static #sigmoid(x) {
        return 1 / (1 + Math.exp(-1 * x));
    }

    static #sigmoid_derivative(y) {
        return y * (1 - y);
    }

    static #relu(x) {
        if (x >= 0) {
            return x;
        }
        return 0;

    }

    static #relu_derivative(y) {
        if (y > 0) {
            return 1;
        }
        return 0;
    }
}

/**
 * Neural network layer
 */
export class Layer {
    static INPUT = 1
    static HIDDEN = 2
    static OUTPUT = 3
    /**
     * @type {Matrix}
     */
    layerError
    /**
     * @type {Matrix}
     */
    weights

    /**
     * @type {Matrix}
     */
    outputs
    /**
     * Constructor
     * @param {int} inputSize
     * @param {int}outputSize
     * @param {number} activation
     * @param {number} layerType
     */
    constructor(inputSize, outputSize, activation, layerType) {
        this.layerType = layerType;
        this.activationFun = Activation.create(activation);
        this.weights = Matrix.randomize(outputSize, inputSize);
        this.biases = Matrix.randomize(outputSize, 1);
        this.inputs = new Array(inputSize);
    }

    /**
     * Feed forward the input matrix to the layer
     * @param {Matrix} input_array - Array of input values
     * @returns {Matrix} - Array of output values
     */
    processFeedForward(input) {
        if (this.layerType == Layer.INPUT) {
            this.inputs = input.data;
            this.outputs = input;
            return input;
        }
        this.inputs = input.data
        let output = Matrix.multiply(this.weights, input);
        output.add(this.biases);
        output.map(this.activationFun.activation);
        this.outputs = output
        return output
    }

    /**
     * Calculate the loss for the layer using the MSE loss function
     * @param target_matrix
     * @param prevLayer
     * @return {*}
     */
    calculateErrorLoss(target_matrix, prevLayer) {
        if (this.layerType == Layer.OUTPUT) {
            this.layerError = Matrix.add(target_matrix, Matrix.multiply(this.outputs, -1));
            return this.layerError;
        }
        const weightTranspose = Matrix.transpose(prevLayer.weights);
        this.layerError = Matrix.multiply(weightTranspose, prevLayer.layerError);
        return this.layerError;
    }

    /**
     * Update the weights of the layer
     */
    updateWeights(nextLayerOutput) {
        //Calculating delta weights
        const nextLayerOutputTransposed = Matrix.transpose(nextLayerOutput);
        const nextWeightsDelta = Matrix.multiply(this.gradient, nextLayerOutputTransposed);

        //Updating weights and biases
        this.weights.add(nextWeightsDelta);
        this.biases.add(this.gradient);
    }


    /**
     * Calculate the gradient of the layer
     * @param {float}learningRate
     */
    calculateGradient(learningRate) {
        this.gradient = Matrix.map(this.outputs, this.activationFun.derivative);
        this.gradient.multiply(this.layerError);
        this.gradient.multiply(learningRate);
    }
}

/**
 * Return the index of the highest value in the array
 * (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
 * @param arr
 * @return {number}
 */
export function argMax(arr) {
    return arr.indexOf(Math.max(...arr));
}
