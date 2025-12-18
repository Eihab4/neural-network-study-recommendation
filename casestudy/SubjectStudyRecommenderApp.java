package casestudy;

import nn.core.Layer;
import nn.core.NeuralNetwork;
import nn.activation.ReLU;
import nn.activation.Linear;
import nn.loss.MeanSquaredError;
import nn.data.DataNormalizer;
import nn.data.DataSplitter;
import nn.data.DataSplitter.SplitResult;
import nn.data.handlers.SkipRowHandler;
import nn.training.Trainer;
import nn.training.TrainingConfig;

public class SubjectStudyRecommenderApp {

    public static void main(String[] args) {
        System.out.println("===========================================");
        System.out.println("   Subject Study Time Recommender");
        System.out.println("===========================================\n");

        try {
            System.out.println("Loading data...");
            DatasetLoader loader = new DatasetLoader(new SkipRowHandler());
            loader.load("casestudy/data/study_time_recommendation_dataset.csv", 5, 1);

            double[][] rawInputs = loader.getInputs();
            double[][] rawExpected = loader.getExpected();
            System.out.println("Loaded " + loader.getSampleCount() + " samples with 5 features.\n");

            System.out.println("Normalizing data...");
            DataNormalizer normalizer = new DataNormalizer();
            normalizer.fit(rawInputs, rawExpected);
            double[][] normalizedInputs = normalizer.normalizeInputs(rawInputs);
            double[][] normalizedExpected = normalizer.normalizeOutputs(rawExpected);

            System.out.println("Splitting data (80% train, 20% test)...");
            DataSplitter splitter = new DataSplitter();
            SplitResult data = splitter.split(normalizedInputs, normalizedExpected, 0.8, true, 42);
            System.out.println("Train samples: " + data.trainInputs.length +
                    ", Test samples: " + data.testInputs.length + "\n");

            System.out.println("Creating neural network...");
            System.out.println("Architecture: 5 -> 8 (ReLU) -> 1 (Linear)\n");
            NeuralNetwork network = new NeuralNetwork(new MeanSquaredError());
            network.addLayer(new Layer(5, 8, new ReLU()));
            network.addLayer(new Layer(8, 1, new Linear()));

            System.out.println("Training...");
            TrainingConfig config = new TrainingConfig()
                    .setLearningRate(0.01)
                    .setEpochs(1000)
                    .setPrintEvery(200)
                    .setVerbose(true);

            Trainer trainer = new Trainer(network, config);
            trainer.train(data.trainInputs, data.trainExpected);

            System.out.println("\nEvaluating on test set...");
            double testLoss = trainer.evaluate(data.testInputs, data.testExpected);
            System.out.println("Test Loss: " + String.format("%.6f", testLoss) + "\n");

            System.out.println("===========================================");
            System.out.println("   Demo Predictions");
            System.out.println("===========================================\n");

            double[][] demoStudents = {
                    { 4.0, 6.5, 0.7, 72, 0.4 },
                    { 2.0, 8.0, 0.3, 85, 0.2 },
                    { 1.0, 5.0, 0.9, 50, 0.8 }
            };

            String[] descriptions = {
                    "studied 4h, slept 6.5h, difficulty 0.7, quiz 72, stress 0.4",
                    "studied 2h, slept 8h, difficulty 0.3, quiz 85, stress 0.2",
                    "studied 1h, slept 5h, difficulty 0.9, quiz 50, stress 0.8"
            };

            for (int i = 0; i < demoStudents.length; i++) {
                double[] normalizedInput = normalizer.normalizeInput(demoStudents[i]);
                double[] prediction = network.predict(normalizedInput);
                double hours = normalizer.denormalizeOutputValue(prediction[0], 0);

                System.out.println("Student " + (i + 1) + ": " + descriptions[i]);
                System.out.println("  -> Recommended study time: " + String.format("%.1f", hours) + " hours\n");
            }

            System.out.println("===========================================");
            System.out.println("   Training Complete!");
            System.out.println("===========================================");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
