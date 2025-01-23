function sigmoid(x) {
  return tf.sigmoid(x);
}

function sigmoidGradient(sigmoidValue) {
  return sigmoidValue.mul(tf.scalar(1).sub(sigmoidValue));
}

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    // Initialize weights and biases as tensors
    this.weights_input_hidden = tf.randomUniform(
      [this.hidden_nodes, this.input_nodes],
      -1,
      1
    );
    this.weights_hidden_output = tf.randomUniform(
      [this.output_nodes, this.hidden_nodes],
      -1,
      1
    );

    this.bias_hidden = tf.randomUniform([this.hidden_nodes, 1], -1, 1);
    this.bias_output = tf.randomUniform([this.output_nodes, 1], -1, 1);

    this.learningRate = 0.1;

    console.log("Weights Input-Hidden:", this.weights_input_hidden.shape);
    console.log("Weights Hidden-Output:", this.weights_hidden_output.shape);
    console.log("Bias Hidden:", this.bias_hidden.shape);
    console.log("Bias Output:", this.bias_output.shape);
    console.log("Inputs Shape:", input_nodes.shape);
  }

  feedForward(input_array) {
    return tf.tidy(() => {
      if (input_array.length !== this.input_nodes) {
        throw new Error(
          `Input array length (${input_array.length}) does not match the number of input nodes (${this.input_nodes}).`
        );
      }

      const inputs = tf.tensor2d(input_array, [this.input_nodes, 1]);

      // Debugging tensor shapes
      console.log("Weights Input-Hidden:", this.weights_input_hidden.shape);
      console.log("Weights Hidden-Output:", this.weights_hidden_output.shape);
      console.log("Inputs Shape:", inputs.shape);

      // Hidden layer activations: matMul(weights_input_hidden, inputs)
      const hidden = sigmoid(
        tf.add(tf.matMul(this.weights_input_hidden, inputs), this.bias_hidden)
      );

      // Debugging shape of hidden tensor
      console.log("Hidden Shape:", hidden.shape);

      // Output layer activations: matMul(weights_hidden_output, hidden)
      const outputs = sigmoid(
        tf.add(tf.matMul(this.weights_hidden_output, hidden), this.bias_output)
      );

      // Debugging output tensor shape
      console.log("Output Shape:", outputs.shape);

      return outputs.arraySync(); // Convert to array for final output
    });
  }

  train(input_array, target_array) {
    tf.tidy(() => {
      // Convert inputs and targets to tensors
      const inputs = tf.tensor2d(input_array, [this.input_nodes, 1]); // Ensure correct shape
      const targets = tf.tensor2d(target_array, [this.output_nodes, 1]); // Ensure correct shape

      // Forward pass
      const hidden = sigmoid(
        tf.add(tf.matMul(this.weights_input_hidden, inputs), this.bias_hidden)
      );
      const outputs = sigmoid(
        tf.add(tf.matMul(this.weights_hidden_output, hidden), this.bias_output)
      );

      // Compute output errors
      const output_errors = targets.sub(outputs);

      // Compute gradients for output layer
      const gradients = sigmoidGradient(outputs)
        .mul(output_errors)
        .mul(this.learningRate);
      const weights_hidden_output_deltas = tf.matMul(
        gradients,
        hidden.transpose() // Transpose hidden for correct shape in matMul
      );

      // Update weights and biases for output layer
      this.weights_hidden_output = this.weights_hidden_output.add(
        weights_hidden_output_deltas
      );
      this.bias_output = this.bias_output.add(gradients);

      // Compute hidden layer errors
      const hidden_errors = tf.matMul(
        this.weights_hidden_output.transpose(),
        output_errors
      );

      // Compute gradients for hidden layer
      const hidden_gradients = sigmoidGradient(hidden)
        .mul(hidden_errors)
        .mul(this.learningRate);
      const weights_input_hidden_deltas = tf.matMul(
        hidden_gradients,
        inputs.transpose() // Transpose inputs for correct shape in matMul
      );

      // Update weights and biases for hidden layer
      this.weights_input_hidden = this.weights_input_hidden.add(
        weights_input_hidden_deltas
      );
      this.bias_hidden = this.bias_hidden.add(hidden_gradients);

      // Dispose tensors after use to manage memory

    });

    // Log weights and bias shapes to monitor updates
    console.log("Weights Input-Hidden:", this.weights_input_hidden.shape);
    console.log("Weights Hidden-Output:", this.weights_hidden_output.shape);
    console.log("Bias Hidden:", this.bias_hidden.shape);
    console.log("Bias Output:", this.bias_output.shape);
  }

  setLearningRate(learning_rate = 0.1) {
    this.learningRate = learning_rate;
  }
}

let nn = new NeuralNetwork(2, 2, 1);
console.log(nn.feedForward([1, 0]));
console.log(nn);

const training_data = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] },
];

function getRandomTrainingData(data) {
  // Generate a random index between 0 and data.length - 1
  const randomIndex = Math.floor(Math.random() * data.length);
  // Return the random object
  return data[randomIndex];
}

// Explicitly set TensorFlow.js backend to CPU (or WebGL if using GPU)
tf.setBackend("cpu"); // You can change this to 'webgl' if using GPU

// Ensure TensorFlow.js is ready before executing the neural network code
tf.ready().then(() => {
  // You can also check which backend is being used
  console.log("Backend being used:", tf.backend());

  let nn = new NeuralNetwork(2, 2, 1);

  // Example training loop (one iteration for simplicity)
  for (let i = 0; i < 1; i++) {
    // const data = getRandomTrainingData(training_data);

    tf.tidy(() => {
      const input_array = [0, 0]; // Shape: [2]
      const target_array = [0]; // Shape: [1]

      // const inputTensor = tf.tensor2d(input_array, [input_array.length, 1]);
      // const targetTensor = tf.tensor2d(target_array, [target_array.length, 1]);

      // nn.train(input_array, target_array);

      // Dispose tensors manually
      // inputTensor.dispose();
      // targetTensor.dispose();
    });
  }

  // Perform feed-forward operations and print outputs
  tf.tidy(() => {
    console.log("Feed Forward Output for [0, 0]:", nn.feedForward([0, 0]));
    console.log("Feed Forward Output for [1, 1]:", nn.feedForward([1, 1]));
    console.log("Feed Forward Output for [1, 0]:", nn.feedForward([1, 0]));
    console.log("Feed Forward Output for [0, 1]:", nn.feedForward([0, 1]));
  });

  // Monitor memory usage after training and feed-forward calls
  console.log(`Number of tensors: ${tf.memory().numTensors}`);
});
