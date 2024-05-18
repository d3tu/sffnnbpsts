export class Layer {
  weights: number[][]
  biases: number[]
  outputs: number[]

  constructor(readonly inputSize: number, readonly outputSize: number) {
    this.weights = Array.from({ length: this.outputSize }, () => {
      return Array.from({ length: this.inputSize }, () => {
        return Math.random() * 2 - 1
      })
    })

    this.biases = Array.from({ length: this.outputSize }, () => {
      return Math.random() * 2 - 1
    })

    this.outputs = new Array(this.outputSize)
  }

  forward(inputs: number[]) {
    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.biases[i]

      for (let j = 0; j < this.inputSize; j++) {
        sum += this.weights[i][j] * inputs[j]
      }

      this.outputs[i] = this.normalActivation(sum)
    }

    return this.outputs
  }

  backward(inputs: number[], errors: number[], learningRate: number) {
    const nextErrors = new Array<number>(this.inputSize)
    
    for (let i = 0; i < this.outputSize; i++) {
      const calculatedError = this.derivateActivation(this.outputs[i]) * errors[i] * learningRate

      this.biases[i] += calculatedError

      for (let j = 0; j < this.inputSize; j++) {
        this.weights[i][j] += calculatedError * inputs[j]

        nextErrors[j] ??= 0
        nextErrors[j] += this.weights[i][j] * errors[i]
      }
    }
    
    return nextErrors
  }

  normalActivation(x: number) {
    return 1 / (1 + Math.exp(-x))
  }

  derivateActivation(x: number) {
    return x * (1 - x)
  }
}

export default class Network {
  layers: Layer[]

  learningRate = 0.1

  constructor(inputSize: number, hiddenSizes: number[], outputSize: number) {    
    this.layers = new Array(hiddenSizes.length + 1)

    let lastSize = inputSize

    for (let i = 0; i < hiddenSizes.length; i++) {
      this.layers[i] = new Layer(lastSize, hiddenSizes[i])

      lastSize = hiddenSizes[i]
    }

    this.layers[hiddenSizes.length] = new Layer(lastSize, outputSize)
  }

  predict(inputs: number[]) {
    let outputs = inputs

    for (const layer of this.layers) {
      outputs = layer.forward(outputs)
    }

    return outputs
  }

  backward(inputs: number[], targets: number[]) {
    const outputs = this.predict(inputs)

    let errors = new Array<number>(outputs.length)

    for (let i = 0; i < outputs.length; i++) {
      errors[i] = targets[i] - outputs[i]
    }

    for (let i = this.layers.length - 1; i >= 0; i--) {
      errors = this.layers[i].backward(i > 0 ? this.layers[i - 1].outputs : inputs, errors, this.learningRate)
    }
  }

  error(inputs: number[], targets: number[]) {
    const outputs = this.predict(inputs)

    let sum = 0

    for (let i = 0; i < outputs.length; i++) {
      sum += (outputs[i] - targets[i]) ** 2
    }

    return sum / 2
  }

  train(dataset: { inputs: number[], targets: number[] }[], epochs = 1000) {
    let error = 0

    for (let i = 0; i < epochs; i++) {
      error = 0

      for (const { inputs, targets } of dataset) {
        this.backward(inputs, targets)

        error += this.error(inputs, targets)
      }

      error /= dataset.length
    }

    return error
  }
}
