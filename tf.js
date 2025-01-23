let model;
let canvas;
let resolution = 20;
let cols;
let rows;
let xs;
function setup() {
  model = tf.sequential();

  let hidden = tf.layers.dense({
    inputShape: [2],
    units: 2,
    activation: "sigmoid",
  });
  let output = tf.layers.dense({
    units: 1,
    activation: "sigmoid",
  });
  model.add(hidden);
  model.add(output);

  const optimizer = tf.train.adam(0.1);
  model.compile({
    optimizer: optimizer,
    loss: "meanSquaredError",
  });

  setTimeout(train, 100);
  canvas = createCanvas(500, 500);
  cols = width / resolution;
  rows = height / resolution;

  let inputs = [];
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let p = i / cols;
      let q = j / rows;
      inputs.push([p, q]);
    }
  }
  xs = tf.tensor2d(inputs);
}

const training_xs = tf.tensor2d([
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
]);
const training_ys = tf.tensor2d([[0], [1], [1], [0]]);
let isTraining = false; // Flag to ensure only one training process at a time

async function draw() {
  if (!isTraining) {
    await two_DRender();
  }
}

async function two_DRender() {
  //   background(0);

  // Only train if not already training
  if (!isTraining) {
    isTraining = true; // Set the flag to true before training

    isTraining = false; // Reset flag after training
  }

  // Predict after training
  tf.tidy(() => {
    let y = model.predict(xs).dataSync();
    let index = 0;

    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        fill(y[index] * 255); // Map predictions to a color scale
        rectMode(CENTER);
        rect(i * resolution, j * resolution, resolution, resolution);
        index++;
      }
    }
  });
}

function train() {
  trainModel().then((result) => {
    // console.log(result.history.loss[0]);
    setTimeout(train, 100);
  });
}

async function trainModel() {
    // Training the model and ensuring memory is cleaned up after each training
    const result = await model.fit(training_xs, training_ys, {
      shuffle: true,
      epochs: 130,
    });
  
    // Explicitly clean up tensors after training
  
    return result;
  }