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
            this.layerNodesCounts.push(layer.inputs.length);
            if (layer.layerType == Layer.OUTPUT) {
                this.layerNodesCounts.push(layer.outputs.length);
            }
        }
    }

    /**
     * Load the pre trained weights from a JSON object
     * @param {NeuralNetwork} dict
     * @returns {NeuralNetwork}
     */
    loadWeights(dict) {
        for (const i in this.layers) {
            this.layers[i].loadWeights(dict.layers[i])
        }
    }

    /**
     * Return the trained weights in a JSON object
     * @returns {Object}
     */
    getWeights() {
        const layers = []
        for (const layersKey in this.layers) {
            layers.push(this.layers[layersKey].getWeights())
        }
        return {
            layerNodesCounts: this.layerNodesCounts,
            layers: layers,
        }
    }

    /**
     * Save the model weights to local storage
     * @param {String} key - the local storage key to save the model weights to
     */
    save(key = "brain") {
        console.log("Saving brain to local storage");
        localStorage.setItem(key, JSON.stringify(this.getWeights()));
    }

    /**
     * Perform the feed foward operation
     * @param {Array} input_array - Array of input values
     * @param {Boolean} GET_ALL_LAYERS - if we need all layers after feed forward instead of just output layer
     * @returns {Array} - the Neural net output for each layer
     */
    feedForward(input_array, GET_ALL_LAYERS = false) {
        this.#feedforwardArgsValidator(input_array)
        let inputMat = Matrix.fromArray(input_array)
        let outputs = [];
        for (let i = 0; i < this.layerNodesCounts.length; i++) {
            outputs[i] = this.layers[i].feedForward(inputMat);
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
        this.feedForward(input, true);
        this.calculateLoss(target);
        this.updateWeights();
    }

    evaluate(inputs, targets) {
        let total = 0
        for (let i = 0; i < inputs.length; i++) {
            const output = this.predict(inputs[i]);
            total += argMax(output) == argMax(targets[i]) ? 1 : 0;
        }
        return total / inputs.length;
    }

    /**
     * Trains with back propagation
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

    // Activation functions
    static #softmax_derivative(y) {
        return y * (1 - y);
    };

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
    layerError

    constructor(inputSize, outputSize, activation, layerType) {
        this.layerType = layerType;
        let weights = new Matrix(outputSize, inputSize);
        weights.randomize()

        let bias = new Matrix(outputSize, 1);
        bias.randomize()

        this.activationFun = Activation.create(activation);
        this.weights = weights;
        this.biases = bias;
        this.inputs = new Array(inputSize);
        this.outputs = new Array(outputSize);

    }

    loadWeights(trainedLayer) {
        this.weights.data = trainedLayer.weights
        this.biases.data = trainedLayer.biases;
        this.outputs.data = trainedLayer.outputs;
        this.inputs = trainedLayer.inputs;
    }

    getWeights() {
        return {
            weights: this.weights.data,
            biases: this.biases.data,
            outputs: this.outputs.data,
            inputs: this.inputs,
            layerType: this.layerType,
        }
    }

    /**
     * Feed forward the input matrix to the layer
     * @param {Array} input_array - Array of input values
     * @param {Boolean} GET_ALL_LAYERS - if we need all layers after feed forward instead of just output layer
     */
    feedForward(input) {
        if (this.layerType == Layer.INPUT) {
            this.inputs = input.data;
            this.outputs = input;
            return input;
        }
        this.inputs = input.data
        input = Matrix.multiply(this.weights, input);
        input.add(this.biases);
        input.map(this.activationFun.activation);
        this.outputs = input
        return input
    }

    calculateErrorLoss(target_matrix, prevLayer) {
        if (this.layerType == Layer.OUTPUT) {
            this.layerError = Matrix.add(target_matrix, Matrix.multiply(this.outputs, -1));
            return this.layerError;
        }
        const weightTranspose = Matrix.transpose(prevLayer.weights);
        this.layerError = Matrix.multiply(weightTranspose, prevLayer.layerError);
        return this.layerError;
    }

    updateWeights(nextLayerOutput) {
        //Calculating delta weights
        const nextLayerOutputTransposed = Matrix.transpose(nextLayerOutput);
        const nextWeightsDelta = Matrix.multiply(this.gradient, nextLayerOutputTransposed);

        //Updating weights and biases
        this.weights.add(nextWeightsDelta);
        this.biases.add(this.gradient);
    }


    calculateGradient(learningRate) {
        this.gradient = Matrix.map(this.outputs, this.activationFun.derivative);
        this.gradient.multiply(this.layerError);
        this.gradient.multiply(learningRate);
    }
}


export function argMax(arr) {
    return arr.indexOf(Math.max(...arr));
}
