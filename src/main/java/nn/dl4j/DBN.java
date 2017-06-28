package nn.dl4j;

import model.Config;
import model.Language;
import nlp.model.Model;
import nlp.model.Pan15BagOfWords;
import nlp.model.Pan15Doc2Vec;
import nlp.model.Pan15Tweet2Vec;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.EarlyStoppingParallelTrainer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;

public class DBN {

    private static Logger log = LoggerFactory.getLogger(DBN.class);
    private MultiLayerNetwork model;
    private static int iterations = 10;
    private static int seed = 42;

    private Language language = Language.ENGLISH;
    private File testFile = new File(Config.PATH + "/english/english-test-short.csv");
    private File trainFile = new File(Config.PATH +"/english/english-train-short.csv");

    private boolean modelFromJson = true;

    public DBN(Language language, String testFile, String trainFile, boolean modelFromJson) {
        this.language = language;
        this.testFile = new File(testFile);
        this.trainFile = new File(trainFile);
        this.modelFromJson = modelFromJson;
    }

    public void trainWithEarlyStopping() throws Exception {
        log.info("TRAINING WITH EARLY STOPPING");
        log.info("Load data from " + trainFile.toString() );
        if (modelFromJson) {
            this.model = getModelFromJson();
        } else {
            this.model = getModel(Config.MODEL.getVecLength());
        }

        model.conf().setPretrain(false);
        RecordReader recordReader = new CSVRecordReader(1);
        recordReader.initialize(new FileSplit(trainFile));

        DataSetIterator iter = new Pan15DataSetIterator(recordReader, language, Config.MODEL);

        model.init();
        log.info("Pretrain model....");
        if (language == Language.ENGLISH || language == Language.SPANISH) {
            INDArray[] pretrainingData = preparePretrainData(language);
            for (INDArray batch : pretrainingData)  model.pretrain(batch);
        }
        log.info("Train model....");

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                        .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(10))
                        .scoreCalculator(new DataSetLossCalculator(iter, true))
                        .evaluateEveryNEpochs(2).modelSaver(saver).build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer =
                new EarlyStoppingParallelTrainer<>(esConf, model, iter, null, 2, 6, 1);
//        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,model,iter);
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());

        model = result.getBestModel();

        log.info("Training done.");

        recordReader = new CSVRecordReader(1);
        log.info("Load verification data from " + testFile.toString() ) ;
        recordReader.initialize(new FileSplit(testFile));
        iter = new Pan15DataSetIterator(recordReader,Config.BATCH_SIZE / 5, language, Config.MODEL);

        RegressionEvaluation eval = new RegressionEvaluation( Config.NUM_OUTPUTS );
        while(iter.hasNext()) {
            DataSet ds = iter.next();
            ds.shuffle();
            INDArray output = model.output(ds.getFeatureMatrix());
            eval.eval(ds.getLabels(), output);
        }
        log.info("Testing done");
        System.out.println(eval.stats());

        try{
            String date = new SimpleDateFormat("yyyy-MM-dd-HHmmssSSS").format(new Date());
            PrintWriter writer = new PrintWriter(language.getName() + date + "-es.txt", "UTF-8");
            writer.println(eval.stats());
            writer.println();
            writer.println(model.conf().toString());
            writer.close();
        } catch (IOException e) {
            log.info("Error saving to file");
        }

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
                .regularization(true)
                .dropOut(Config.DROPOUT)
                .updater(Config.UPDATER)
                .adamMeanDecay(0.5)
                .adamVarDecay(0.5)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(numInputs).nOut(2750).dropOut(0.75)
                        .activation(Activation.RELU).build())
                .layer(1, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
                        .nIn(2750).nOut(2000)
                        .activation(Activation.RELU).build())
                .layer(2, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
                        .nIn(2000).nOut(1000)
                        .activation(Activation.RELU).build())
                .layer(3, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
                        .nIn(1000).nOut(200)
                        .activation(Activation.RELU).build())
                .layer(4, new OutputLayer.Builder(Config.LOSS_FUNCTION)
                        .nIn(200).nOut(Config.NUM_OUTPUTS).updater(Config.UPDATER)
                        .adamMeanDecay(0.6).adamVarDecay(0.7)
                        .build())
                .pretrain(true).backprop(true)
                .build();
        return new MultiLayerNetwork(conf);
    }

    private MultiLayerNetwork getModelFromJson() throws IOException {
        String path = "./src/main/resources/models/d2v/" + language.getName() + "-model.json";
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        String json =  new String(encoded, StandardCharsets.UTF_8);
        System.out.println(json);
        return new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));

    }

    INDArray[] preparePretrainData(Language language) {
        int slices = 8;
        List<INDArray> indArrays = getData(language);
        Collections.shuffle(indArrays);
        INDArray[] pretrainData = new INDArray[slices];
        int n = indArrays.size()/slices;
        System.out.println(indArrays.size());
        for(int i = 0, j = 0; i < slices && j < indArrays.size(); i++) {
            System.out.println(j + " " + (j+n));
            if (j+n < indArrays.size()) pretrainData[i] = Nd4j.vstack(indArrays.subList(j, j+n));
            else pretrainData[i] = Nd4j.vstack(indArrays.subList(j, indArrays.size()));
            j = j+n;
        }
        return pretrainData;
    }

    List<INDArray> getData(Language language) {
        Collection<INDArray> values = null;
        if (Config.MODEL instanceof  Pan15Tweet2Vec) {
            Pan15Tweet2Vec p = new Pan15Tweet2Vec("./src/main/resources/tweet2vec/pretr/");
            values = p.parseLanguage(language).values();
        } else if (Config.MODEL instanceof Pan15Doc2Vec) {
            Pan15Doc2Vec p = new Pan15Doc2Vec("./src/main/resources/doc2vec/pretr/");
            values = p.parseLanguage(language).values();
        } else if (Config.MODEL instanceof Pan15BagOfWords) {
            values = ((Pan15BagOfWords) Config.MODEL).parseLanguage(language);
        }
        return new ArrayList<>(values);
    }


    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

}
