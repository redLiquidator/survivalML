package com.example.demo;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.example.demo.TitanicSurvival.Util.SparkSessionUtil;
import com.example.demo.TitanicSurvival.Util.Util;
import com.example.demo.TitanicSurvival.Util.Util.VectorPair;

public class SurvivalPredictionMLP {
	
    public static void main(String[] args) {
    	
        SparkSession spark = SparkSessionUtil.getInstance();
        
        Dataset<Row> trainingDF = Util.getTrainingDF();

        MultivariateStatisticalSummary summary = Util.summary;
        double meanFare = summary.mean().apply(0);
        double meanAge = summary.mean().apply(1);     

        /*
        Map<String, Object> m = new HashMap<String, Object>();
        m.put("Age", meanAge);
        m.put("Fare", meanFare);
        Dataset<Row> trainingDF2 = trainingDF.na().fill(m);
        trainingDF2.show();      
        */  
        
        Vector stddev = Vectors.dense(Math.sqrt(summary.variance().apply(0)), Math.sqrt(summary.variance().apply(1)));
        Vector mean = Vectors.dense(summary.mean().apply(0), summary.mean().apply(1));
        StandardScalerModel scaler = new StandardScalerModel(stddev, mean);

        // The columns of a row in the result can be accessed by field index
        Encoder<Integer> integerEncoder = Encoders.INT();
        Encoder<Double> doubleEncoder = Encoders.DOUBLE();
        Encoders.BINARY();
        Encoder<Vector> vectorEncoder = Encoders.kryo(Vector.class);
        Encoders.tuple(integerEncoder, vectorEncoder);
        Encoders.tuple(doubleEncoder, vectorEncoder);

        JavaRDD<VectorPair> scaledRDD = trainingDF.toJavaRDD().map(row -> {
                VectorPair vectorPair = new VectorPair();
                vectorPair.setLable(new Double(row.<Integer>getAs("Survived")));
                vectorPair.setFeatures(Util.getScaledVector(
                                row.<Double>getAs("Fare"),
                                row.<Double>getAs("Age"),
                                row.<Integer>getAs("Pclass"),
                                row.<Integer>getAs("Sex"),
                                row.isNullAt(7) ? 0d : row.<Integer>getAs("Embarked"),
                                scaler));

                return vectorPair;
        });

        Dataset<Row> scaledDF = spark.createDataFrame(scaledRDD, VectorPair.class);
        scaledDF.show();

        Dataset<Row> scaledData2 = MLUtils.convertVectorColumnsToML(scaledDF);
        
        Dataset<Row> data = scaledData2.toDF("features", "label");
        Dataset<Row>[] datasets = data.randomSplit(new double[]{0.80, 0.20}, 12345L);
        
        Dataset<Row> trainingData = datasets[0];
        Dataset<Row> validationData = datasets[1];
        
        int[] layers = new int[] {10, 16, 32, 2};
        // create the trainer and set its parameters
        MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setTol(1E-8)
                .setMaxIter(1000);

        MultilayerPerceptronClassificationModel model = mlp.fit(trainingData);
        
        Dataset<Row> predictions = model.transform(validationData);
        predictions.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction");
        MulticlassClassificationEvaluator evaluator1 = evaluator.setMetricName("accuracy");
        MulticlassClassificationEvaluator evaluator2 = evaluator.setMetricName("weightedPrecision");
        MulticlassClassificationEvaluator evaluator3 = evaluator.setMetricName("weightedRecall");
        MulticlassClassificationEvaluator evaluator4 = evaluator.setMetricName("f1");

        // compute the classification accuracy, precision, recall, f1 measure and error on test data.
        double accuracy = evaluator1.evaluate(predictions);
        double precision = evaluator2.evaluate(predictions);
        double recall = evaluator3.evaluate(predictions);
        double f1 = evaluator4.evaluate(predictions);

        // Print the performance metrics
        System.out.println("Accuracy = " + accuracy);
        System.out.println("Precision = " + precision);
        System.out.println("Recall = " + recall);
        System.out.println("F1 = " + f1);
        System.out.println("Test Error = " + (1 - accuracy));
        
        Dataset<Row> testDF = Util.getTestDF();
        testDF.show();   
        
        Map<String, Object> m = new HashMap<String, Object>();
        m.put("Age", meanAge);
        m.put("Fare", meanFare);
        
        Dataset<Row> testDF2 = testDF.na().fill(m);
        testDF2.show();
        
        JavaRDD<VectorPair> testRDD = testDF2.javaRDD().map(row -> {
            VectorPair vectorPair = new VectorPair();
            vectorPair.setLable(row.<Integer>getAs("PassengerId"));
            vectorPair.setFeatures(Util.getScaledVector(
                    row.<Double>getAs("Fare"),
                    row.<Double>getAs("Age"),
                    row.<Integer>getAs("Pclass"),
                    row.<Integer>getAs("Sex"),
                    row.<Integer>getAs("Embarked"),
                    scaler));
            return vectorPair;
        });

        Dataset<Row> scaledTestDF = spark.createDataFrame(testRDD, VectorPair.class);

        Dataset<Row> finalTestDF = MLUtils.convertVectorColumnsToML(scaledTestDF).toDF("features", "PassengerId");
        trainingData.show();
        finalTestDF.show();

        Dataset<Row> resultDF = model.transform(finalTestDF).select("PassengerId", "prediction");  
        resultDF.show();
        resultDF.write().format("com.databricks.spark.csv").option("header", true).save("result/result.csv");
    }
}
