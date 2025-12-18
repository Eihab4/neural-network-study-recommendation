package test;

import nn.activation.*;
import nn.loss.*;
import nn.initialization.*;
import nn.data.handlers.*;

public class ComponentTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("===========================================");
        System.out.println("   Component Tests");
        System.out.println("===========================================\n");

        testActivationFunctions();
        testLossFunctions();
        testWeightInitializers();
        testMissingValueHandlers();

        System.out.println("\n===========================================");
        System.out.println("   Results: " + passed + " passed, " + failed + " failed");
        System.out.println("===========================================");
    }

    private static void testActivationFunctions() {
        System.out.println("--- Activation Functions ---");

        ReLU relu = new ReLU();
        test("ReLU positive", relu.activate(5.0), 5.0);
        test("ReLU negative", relu.activate(-3.0), 0.0);
        test("ReLU zero", relu.activate(0.0), 0.0);
        test("ReLU derivative positive", relu.derivative(5.0), 1.0);
        test("ReLU derivative negative", relu.derivative(-3.0), 0.0);

        Sigmoid sigmoid = new Sigmoid();
        test("Sigmoid 0", sigmoid.activate(0.0), 0.5);
        test("Sigmoid large positive", sigmoid.activate(10.0) > 0.99, true);
        test("Sigmoid large negative", sigmoid.activate(-10.0) < 0.01, true);
        test("Sigmoid derivative at 0", Math.abs(sigmoid.derivative(0.0) - 0.25) < 0.001, true);

        Tanh tanh = new Tanh();
        test("Tanh 0", tanh.activate(0.0), 0.0);
        test("Tanh large positive", tanh.activate(5.0) > 0.99, true);
        test("Tanh large negative", tanh.activate(-5.0) < -0.99, true);
        test("Tanh derivative at 0", tanh.derivative(0.0), 1.0);

        Linear linear = new Linear();
        test("Linear positive", linear.activate(5.0), 5.0);
        test("Linear negative", linear.activate(-3.0), -3.0);
        test("Linear derivative", linear.derivative(100.0), 1.0);

        System.out.println();
    }

    private static void testLossFunctions() {
        System.out.println("--- Loss Functions ---");

        MeanSquaredError mse = new MeanSquaredError();
        double[] pred1 = { 1.0 };
        double[] exp1 = { 1.0 };
        test("MSE perfect prediction", mse.compute(pred1, exp1), 0.0);

        double[] pred2 = { 2.0 };
        double[] exp2 = { 0.0 };
        test("MSE error = 4", mse.compute(pred2, exp2), 4.0);

        double[] grad = mse.gradient(pred2, exp2);
        test("MSE gradient", grad[0], 4.0);

        CrossEntropyLoss ce = new CrossEntropyLoss();
        double[] pred3 = { 0.9 };
        double[] exp3 = { 1.0 };
        test("CrossEntropy near correct", ce.compute(pred3, exp3) < 0.2, true);

        double[] pred4 = { 0.1 };
        double[] exp4 = { 1.0 };
        test("CrossEntropy far wrong > near correct",
                ce.compute(pred4, exp4) > ce.compute(pred3, exp3), true);

        System.out.println();
    }

    private static void testWeightInitializers() {
        System.out.println("--- Weight Initializers ---");

        RandomUniformInitializer uniform = new RandomUniformInitializer(-1, 1, 42);
        double[][] weights1 = uniform.initialize(3, 4);
        test("Uniform shape rows", weights1.length, 3);
        test("Uniform shape cols", weights1[0].length, 4);

        boolean allInRange = true;
        for (double[] row : weights1) {
            for (double w : row) {
                if (w < -1 || w > 1)
                    allInRange = false;
            }
        }
        test("Uniform all in range [-1,1]", allInRange, true);

        HeInitializer he = new HeInitializer(42);
        double[][] weights2 = he.initialize(100, 50);
        test("He shape rows", weights2.length, 100);
        test("He shape cols", weights2[0].length, 50);

        double sum = 0;
        for (double[] row : weights2) {
            for (double w : row) {
                sum += w;
            }
        }
        double mean = sum / (100 * 50);
        test("He mean close to 0", Math.abs(mean) < 0.1, true);

        System.out.println();
    }

    private static void testMissingValueHandlers() {
        System.out.println("--- Missing Value Handlers ---");

        double[] means = { 1.0, 2.0, 3.0 };

        UseZeroHandler zero = new UseZeroHandler();
        test("UseZero returns 0", zero.handle(0, means), 0.0);

        UseMeanHandler mean = new UseMeanHandler();
        test("UseMean column 0", mean.handle(0, means), 1.0);
        test("UseMean column 1", mean.handle(1, means), 2.0);
        test("UseMean column 2", mean.handle(2, means), 3.0);

        SkipRowHandler skip = new SkipRowHandler();
        test("SkipRow returns null", skip.handle(0, means) == null, true);

        ThrowErrorHandler error = new ThrowErrorHandler();
        boolean threw = false;
        try {
            error.handle(0, means);
        } catch (RuntimeException e) {
            threw = true;
        }
        test("ThrowError throws exception", threw, true);

        System.out.println();
    }

    private static void test(String name, double actual, double expected) {
        if (Math.abs(actual - expected) < 0.0001) {
            System.out.println("  ✓ " + name);
            passed++;
        } else {
            System.out.println("  ✗ " + name + " (expected " + expected + ", got " + actual + ")");
            failed++;
        }
    }

    private static void test(String name, int actual, int expected) {
        if (actual == expected) {
            System.out.println("  ✓ " + name);
            passed++;
        } else {
            System.out.println("  ✗ " + name + " (expected " + expected + ", got " + actual + ")");
            failed++;
        }
    }

    private static void test(String name, boolean condition, boolean expected) {
        if (condition == expected) {
            System.out.println("  ✓ " + name);
            passed++;
        } else {
            System.out.println("  ✗ " + name + " (expected " + expected + ", got " + condition + ")");
            failed++;
        }
    }
}
