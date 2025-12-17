# Neural Network Study Recommendation

A from-scratch, generic feedforward neural network library in Java, applied to a math study-time recommendation case study.

## Requirements

- **Java JDK 17+** (tested with Java 24.0.1)

Verify installation:

```bash
java --version
javac --version
```

## Project Structure

```
neural-network-study-recommendation/
├── src/nn/                    # Neural Network Library
│   ├── core/                  # NeuralNetwork, Layer
│   ├── activation/            # ReLU, Sigmoid, Tanh, Linear
│   ├── initialization/        # Weight initializers (Random, He)
│   ├── loss/                  # MSE, CrossEntropy
│   ├── training/              # Trainer, TrainingConfig
│   ├── data/                  # DataSplitter, DataNormalizer
│   └── util/                  # MatrixUtils
│
├── test/                      # Manual test classes
├── casestudy/                 # Math study recommendation app
│   └── data/                  # Dataset CSV file
└── out/                       # Compiled .class files
```

## How to Compile

Compile all source files:

```bash
javac -d out src/nn/**/*.java
```

Or compile specific packages:

```bash
javac -d out src/nn/activation/*.java
javac -d out src/nn/core/*.java
```

## How to Run Tests

After compiling:

```bash
java -cp out test.TestActivationFunctions
java -cp out test.TestNeuralNetwork
```

Or run all tests:

```bash
java -cp out test.TestRunner
```

## How to Run the Case Study

```bash
# Compile case study (after compiling nn library)
javac -cp out -d out casestudy/*.java

# Run the application
java -cp out casestudy.MathStudyRecommenderApp
```

## Features

### Activation Functions

- Sigmoid
- ReLU
- Tanh
- Linear

### Loss Functions

- Mean Squared Error (MSE)
- Cross-Entropy

### Weight Initialization

- Random Uniform
- He Initialization

### Training

- Forward propagation
- Backpropagation with chain rule
- Configurable learning rate, epochs, batch size
- Training loss tracking

## Case Study

Predicts recommended daily study hours for math based on:

- Study hours yesterday
- Sleep hours
- Subject difficulty
- Last quiz score
- Stress level
