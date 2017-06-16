package nn.dl4j;

import model.Config;
import model.Language;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.data.DataSetIteratorProvider;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.layers.RBMLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.multilayer.LocalMultiLayerNetworkSaver;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetRegressionScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class HyperParamOptimization {

    private static int numOutputs = 5;
    private static int numEpochs = 10;
    private static int seed = 42;
    private static int batchSize = 500;
    private static LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.MSE;
    private static Updater updater = Updater.ADAM;

    private static Language language = Language.ENGLISH;
    private static File testFile = new File(Config.PATH + "/english/english-test-pan15.csv");
    private static File trainFile = new File(Config.PATH +"/english/english-train-pan15.csv");

    private static int idxFrom = 2;
    private static int idxTo = 6;
    private static boolean regression = true;

    private static Logger log = LoggerFactory.getLogger(HyperParamOptimization.class);

    public static void main(String[] args) throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().setElementThreshold(16384);
        NativeOpsHolder.getInstance().getDeviceNativeOps().setTADThreshold(64);

        //First: Set up the hyperparameter configuration space.
        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
        ParameterSpace<Integer> layer1out = new IntegerParameterSpace(Config.MODEL.getVecLength(),5000);
        ParameterSpace<Integer> layer2out = new IntegerParameterSpace(Config.MODEL.getVecLength(),5000);
        ParameterSpace<Integer> layer3out = new IntegerParameterSpace(Config.MODEL.getVecLength(),5000);
        ParameterSpace<Integer> layer4out = new IntegerParameterSpace(Config.MODEL.getVecLength(),5000);

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                .iterations(100)
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .regularization(true)
                .dropOut(0.5)
                .updater(updater)
                .adamMeanDecay(0.5)
                .adamVarDecay(0.5)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRateHyperparam)
                .addLayer(new RBMLayerSpace.Builder().hiddenUnit(RBM.HiddenUnit.BINARY).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .dropOut(0.75)
                        .activation("relu")
                        .lossFunction(lossFunction)
                        .nIn(Config.MODEL.getVecLength()).nOut(layer1out)
                        .build())
                .addLayer(new RBMLayerSpace.Builder()
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .activation("relu")
                        .lossFunction(lossFunction)
                        .nIn(layer1out).nOut(layer2out).build())
                .addLayer(new RBMLayerSpace.Builder()
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .activation("relu")
                        .lossFunction(lossFunction)
                        .nIn(layer2out)
                        .nOut(layer3out).build())
                .addLayer(new RBMLayerSpace.Builder()
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .activation("relu")
                        .lossFunction(lossFunction)
                        .nIn(layer3out)
                        .nOut(layer4out).build())
                .addLayer(new OutputLayerSpace.Builder()
                        .lossFunction(lossFunction)
                        .updater(updater)
                        .adamMeanDecay(0.6)
                        .adamVarDecay(0.7)
                        .nIn(layer4out)
                        .nOut(numOutputs).build())
                .pretrain(true).backprop(true).build();


        //Now: We need to define a few configuration options
        // (a) How are we going to generate candidates? (random search or grid search)
        CandidateGenerator<DL4JConfiguration> candidateGenerator = new RandomSearchGenerator<>(hyperparameterSpace);    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);

        // (b) How are going to provide data? For now, we'll use a DataSetIterator
        RecordReader recordReader = new CSVRecordReader(1);
        recordReader.initialize(new FileSplit(trainFile));
        DataSetIterator train = new MultipleEpochsIterator(numEpochs, new Pan15DataSetIterator(recordReader, batchSize, idxFrom, idxTo, regression,  language,   Config.MODEL));

        recordReader = new CSVRecordReader(1);
        recordReader.initialize(new FileSplit(testFile));
        DataSetIterator test = new Pan15DataSetIterator(recordReader,batchSize / 5, idxFrom, idxTo,true, language, Config.MODEL);

        DataProvider<DataSetIterator> dataProvider = new DataSetIteratorProvider(train, test);

        // (c) How we are going to save the models that are generated and tested?
        //     In this example, let's save them to disk the working directory
        //     This will result in examples being saved to arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
        String date = new SimpleDateFormat("yyyy-MM-dd-HHmmssSSS").format(new Date());
        String baseSaveDirectory = "arbiter-" + date + "/";
        File f = new File(baseSaveDirectory);
        if(f.exists()) f.delete();
        f.mkdir();
        ResultSaver<DL4JConfiguration,MultiLayerNetwork,Object> modelSaver = new LocalMultiLayerNetworkSaver<>(baseSaveDirectory);

        // (d) What are we actually trying to optimize?
        //     In this example, let's use classification accuracy on the test set
        ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction = new TestSetRegressionScoreFunction(RegressionValue.RMSE);

        // (e) When should we stop searching? Specify this with termination conditions
        //     For this example, we are stopping the search at 15 minutes or 20 candidates - whichever comes first
        TerminationCondition[] terminationConditions = {new MaxTimeCondition(48, TimeUnit.HOURS), new
                MaxCandidatesCondition(200)};


        //Given these configuration options, let's put them all together:
        OptimizationConfiguration<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object> configuration
                = new OptimizationConfiguration.Builder<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        //And set up execution locally on this machine:
        IOptimizationRunner<DL4JConfiguration,MultiLayerNetwork,Object> runner
                = new LocalOptimizationRunner<>(configuration, new MultiLayerNetworkTaskCreator<>());

        runner.execute();


        //Print out some basic stats regarding the optimization procedure
        StringBuilder sb = new StringBuilder();
        sb.append("Best score: ").append(runner.bestScore()).append("\n")
                .append("Index of model with best score: ").append(runner.bestScoreCandidateIndex()).append("\n")
                .append("Number of configurations evaluated: ").append(runner.numCandidatesCompleted()).append("\n");
        log.info(sb.toString());
        log.info(language.toString());
        log.info(Config.MODEL.getClass().getName());
        log.info(testFile.getPath());
        log.info(trainFile.getPath());


        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference<DL4JConfiguration,MultiLayerNetwork,Object>> allResults = runner.getResults();

        OptimizationResult<DL4JConfiguration,MultiLayerNetwork,Object> bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = bestResult.getResult();
        log.info("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getLayerWiseConfigurations().toJson());


        //Note: UI server will shut down once execution is complete, as JVM will exit
        //So do a Thread.sleep(1 minute) to keep JVM alive, so that network configurations can be viewed
        Thread.sleep(60000);
        System.exit(0);
    }
}
