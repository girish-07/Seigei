const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");

async function load_model() {
    const handler = tfn.io.fileSystem("./model.json");
    let m = await tf.loadLayersModel(handler);
    console.log(m);
    return m;
}

let model = load_model();


model.then(function (res) {
    // The below code is sample snippet, write the functionality here
//     const example = tf.browser.fromPixels(canvas);
//     const prediction = model.predict(example);
//     console.log(prediction);
// }, function (err) {
//     console.log(err);
});