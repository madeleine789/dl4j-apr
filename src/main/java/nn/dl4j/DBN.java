package nn.dl4j;

import model.Config;
import model.Language;
import model.Personality;
import nlp.model.Model;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.concurrent.TimeUnit;

public class DBN  {

    private static Logger log = LoggerFactory.getLogger(DBN.class);
    private MultiLayerNetwork model;
    private static int numOutputs = 5;
    private static int iterations = 100;
    private static int seed = 42;
    private static int listenerFreq = 1000;
    private static int batchSize = 128;
    private static LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
    private static Updater updater = Updater.ADAM;

    private Language language = Language.ENGLISH;
    private File testFile = new File(Config.PATH + "/english/english-test-short.csv");
    private File trainFile = new File(Config.PATH +"/english/english-train-short.csv");
    static String directory = "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/early-stopping/";

    private int idxFrom = 2;
    private int idxTo = 6;
    private boolean regression = true;
    private Model inputModel;

    public DBN(Language language, Model model, String testFile, String trainFile) {
        this.language = language;
        this.inputModel = model;
        this.testFile = new File(testFile);
        this.trainFile = new File(trainFile);
        regression = true;
    }

    public DBN(Language language, Model model, String testFile, String trainFile, Personality label) {
        this(language, model, testFile, trainFile);
        this.idxFrom = label.getIndex();
        this.idxTo = label.getIndex();
        numOutputs = 1;
        regression = true;
    }

    public DBN(Language language, Model model, String testFile, String trainFile, int index) {
        this(language, model, testFile, trainFile);
        this.idxFrom = index;
        this.idxTo = index;
        numOutputs = 1;
        regression = false;
    }

    public void train() throws Exception {

        log.info("Load data from " + trainFile.toString() );
        RecordReader recordReader = new CSVRecordReader(1);
        recordReader.initialize(new FileSplit(trainFile));
        DataSetIterator iter = new Pan15DataSetIterator(recordReader,batchSize, idxFrom, idxTo, regression, language, inputModel);

            log.info("Train model....");
            while(iter.hasNext()) {
                DataSet ds = iter.next();
                model.fit( ds ) ;
            }
            log.info("Training done.");
    }


    public String test() throws Exception {

        RecordReader recordReader = new CSVRecordReader(1);
        log.info("Load verification data from " + testFile.toString() ) ;
        recordReader.initialize(new FileSplit(testFile));
        DataSetIterator iter = new Pan15DataSetIterator(recordReader,batchSize / 4, idxFrom, idxTo,true, language, inputModel);

        RegressionEvaluation eval = new RegressionEvaluation( numOutputs );
        while(iter.hasNext()) {
            DataSet ds = iter.next();
            INDArray predict2 = model.output(ds.getFeatureMatrix(), Layer.TrainingMode.TEST);
            eval.eval(ds.getLabels(), predict2);
        }
        log.info("Testing done");

        return eval.stats() ;
    }

    public static MultiLayerConfiguration getConf(int numInputs) {
        return getModel(numInputs).getLayerWiseConfigurations();
    }

    public static MultiLayerNetwork getModel(int numInputs) {
        MultiLayerConfiguration conf =  new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
//                .regularization(true)
//                .l2(0.001)
//                .dropOut(0.2)
                .updater(updater)
                .adamMeanDecay(0.5)
                .adamVarDecay(0.5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(numInputs).nOut(2750)
                        .activation("relu").lossFunction(lossFunction).build())
                .layer(1, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(2750).nOut(2000)
                        .activation("relu").lossFunction(lossFunction).build())
                .layer(2, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(2000).nOut(1000)
                        .activation("relu").lossFunction(lossFunction).build())
                .layer(3, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(1000).nOut(200)
                        .activation("relu").lossFunction(lossFunction).build())
                .layer(4, new OutputLayer.Builder(lossFunction)
                        .nIn(200).nOut(numOutputs).updater(updater).adamMeanDecay(0.6).adamVarDecay(0.7).build())
                .pretrain(true).backprop(true)
                .build();
        return new MultiLayerNetwork(conf);
    }
//    lossFunction = LossFunctions.LossFunction.MCXENT - multiclass classification
//    lossFunction = LossFunctions.LossFunction.XENT - binary classification
    public MultiLayerNetwork getClassificationModel(int numInputs, LossFunctions.LossFunction lossFunction) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(numInputs).nOut(2700)
                        .activation("relu")
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(1, new RBM.Builder().nIn(2700).nOut(2000)
                        .activation("relu")
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(2, new RBM.Builder().nIn(2000).nOut(1000)
                        .activation("relu")
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(3, new RBM.Builder().nIn(1000).nOut(200)
                        .activation("relu")
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation
                        ("sigmoid")
                        .nIn(200).nOut(numOutputs).build())
                .pretrain(true).backprop(true)
                .build();
        return new MultiLayerNetwork(conf);
    }

    public MultiLayerNetwork trainWithEarlyStopping() throws IOException,
            InterruptedException {
        MultiLayerConfiguration myNetworkConfiguration = getConf(inputModel.getVecLength());

        RecordReader recordReader = new CSVRecordReader(1);
        recordReader.initialize(new FileSplit(testFile));
        DataSetIterator myTestData = new Pan15DataSetIterator(recordReader,100, 2,6,true, language, inputModel);
        recordReader.initialize(new FileSplit(trainFile));
        DataSetIterator myTrainData = new Pan15DataSetIterator(recordReader,500, 2,6,true, language, inputModel);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(300))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(myTestData, true))
                .evaluateEveryNEpochs(5)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,myNetworkConfiguration,myTrainData);

        //Conduct early stopping training:
        EarlyStoppingResult result = trainer.fit();

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //Get the best model:
        return (MultiLayerNetwork) result.getBestModel();
    }


    public void runTrainingAndValidate() {
        if (regression) {
            this.model = getModel(inputModel.getVecLength());
        } else {
            LossFunctions.LossFunction lossFunction = (idxFrom == 7) ?
                    LossFunctions.LossFunction.XENT : LossFunctions.LossFunction.MCXENT;
            this.model = getClassificationModel(inputModel.getVecLength(), lossFunction);
        }

        model.init();
        model.setListeners(Collections.singletonList(new ScoreIterationListener(listenerFreq)));
        try {
            this.train();
            System.out.println(this.test());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }
}
