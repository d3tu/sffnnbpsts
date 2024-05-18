# sffnnbpst
A Simple Feed Forward Neural Network with Back Propagation from Scratch in Typescript

```ts
import Network from "."

const inputSize = 2
const hiddenSizes = [3]
const outputSize = 1

const network = new Network(inputSize, hiddenSizes, outputSize)

const dataset = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] }
]

const epochs = 10_000

console.log("current error:", network.train(dataset, epochs))

for (const { inputs } of dataset) {
  console.log(inputs[0], "xor", inputs[1], "~=", network.predict(inputs))
}
```
