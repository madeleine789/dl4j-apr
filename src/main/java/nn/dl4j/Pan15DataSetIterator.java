package nn.dl4j;

import org.datavec.api.io.WritableConverter;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import model.Language;
import nlp.Pan15SentencePreProcessor;
import nlp.Pan15Word2Vec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Pan15DataSetIterator extends AbstractDataSetIterator {
    protected RecordReader recordReader;
    protected WritableConverter converter;
    protected int batchSize = 10;
    protected int maxNumBatches = -1;
    protected int labelIndex = -1;
    protected int labelIndexTo = -1;
    protected int numPossibleLabels = -1;
    protected DataSet last;
    protected boolean regression = false;
    protected Language language = Language.ENGLISH;
    int maxlen = Pan15Word2Vec.VEC_SIZE;

    private Pan15Word2Vec pan15Word2Vec = new Pan15Word2Vec();
    private Pan15SentencePreProcessor preProcessor = new Pan15SentencePreProcessor();

    /**
     * Main constructor for multi-label regression (i.e., regression with multiple outputs)
     *
     * @param recordReader      RecordReader to get data from
     * @param labelIndexFrom    Index of the first regression target
     * @param labelIndexTo      Index of the last regression target, inclusive
     * @param batchSize         Minibatch size
     * @param regression        Require regression = true. Mainly included to avoid clashing with other constructors previously defined :/
     */
    public Pan15DataSetIterator(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
                                boolean regression, Language language, ElementsLearningAlgorithm<VocabWord> learningAlgorithm){
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndexFrom, labelIndexTo, -1, -1, regression,
                language, learningAlgorithm);
    }

    /**
     * Main constructor for single-label regression (i.e., regression with one outputs)
     *
     * @param recordReader      RecordReader to get data from
     * @param labelIndexFrom    Index of the first regression target
     * @param batchSize         Minibatch size
     * @param regression        Require regression = true. Mainly included to avoid clashing with other constructors previously defined :/
     */
    public Pan15DataSetIterator(RecordReader recordReader, int batchSize, int labelIndexFrom, boolean regression,
                                Language language, ElementsLearningAlgorithm<VocabWord> learningAlgorithm){
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndexFrom, labelIndexFrom,
                -1, -1, regression, language, learningAlgorithm);
    }


    /**
     * Main constructor
     *
     * @param recordReader      the recordreader to use
     * @param converter         the batch size
     * @param maxNumBatches     Maximum number of batches to return
     * @param labelIndexFrom    the index of the label (for classification), or the first index of the labels for multi-output regression
     * @param labelIndexTo      only used if regression == true. The last index _inclusive_ of the multi-output regression
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param regression        if true: regression. If false: classification (assume labelIndexFrom is a
     */
    public Pan15DataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndexFrom,
                                 int labelIndexTo, int numPossibleLabels, int maxNumBatches, boolean regression,
                                 Language language, ElementsLearningAlgorithm<VocabWord> learningAlgorithm) {
        super(recordReader, batchSize, maxNumBatches);
        this.recordReader = recordReader;
        this.converter = converter;
        this.batchSize = batchSize;
        this.maxNumBatches = maxNumBatches;
        this.labelIndex = labelIndexFrom;
        this.labelIndexTo = labelIndexTo;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
        this.language = language;
        this.pan15Word2Vec = new Pan15Word2Vec(learningAlgorithm);
    }

    public DataSet getDataSet(List<Writable> record) {
        List<Writable> currList;
        if (record instanceof List)
            currList = record;

        else
            currList = new ArrayList<>(record);
        //allow people to specify label index as -1 and infer the last possible label

        INDArray label = null;
        INDArray featureVector = null;
        int labelCount = 0;
        if (labelIndexTo != labelIndex) {
            for (int j = 1; j <= labelIndexTo; j++) {
                Writable current = currList.get(j);

                if (regression && j >= labelIndex && j <= labelIndexTo) {
                    //This is the multi-label regression case
                    if (label == null) label = Nd4j.create(1, (labelIndexTo - labelIndex + 1));
                    label.putScalar(labelCount++, current.toDouble());
                } else {
                    String value = preProcessor.preProcess(current.toString().substring(1,current.toString().lastIndexOf('\"')));
                    if (featureVector == null) {
                        if(pan15Word2Vec.getWordEmbeddings(value, language).stream().noneMatch(Objects::nonNull))
                            return new DataSet(Nd4j.zeros(1, maxlen),Nd4j.zeros(1, (labelIndexTo - labelIndex + 1))) ;

                        featureVector = pan15Word2Vec.getSentence2VecSum(value, language);
                    }
                }
            }
            return new DataSet(featureVector, label);
        } else {
            Writable current = currList.get(labelIndex);
            label = Nd4j.create(1, 1);
            label.putScalar(labelIndex, current.toDouble());
            current = currList.get(1);
            String value = preProcessor.preProcess(current.toString().substring(1,current.toString().lastIndexOf('\"')));
            if(pan15Word2Vec.getWordEmbeddings(value, language).stream().noneMatch(Objects::nonNull))
                return new DataSet(Nd4j.zeros(1, maxlen), Nd4j.zeros(1, (labelIndexTo - labelIndex + 1))) ;
            featureVector = pan15Word2Vec.getSentence2VecSum(value, language);
        }

        return new DataSet(featureVector, label);
    }
}
