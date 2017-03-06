package nlp;

import model.Language;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.GloVe;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import parsing.pan15.Pan15Author;
import parsing.pan15.Pan15Parser;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import static nlp.Utils.getSentencesFromLanguage;

public class Pan15Word2Vec implements Model {

    public static final Pan15SentencePreProcessor PREPROCESSOR = new Pan15SentencePreProcessor();
    private static HashMap<Language, Word2Vec> languageWord2VecMap = new HashMap<>();
    private TokenizerFactory t = new DefaultTokenizerFactory();
    private HashMap<Language, HashMap<String, Pan15Author>> languages = Utils.getLanguages();
    private static final Integer VEC_LENGTH = 250;
    private ElementsLearningAlgorithm<VocabWord> learningAlgorithm = new CBOW<>();

    public Pan15Word2Vec() {
        getW2VFromFile();
    }

    public Pan15Word2Vec(ElementsLearningAlgorithm<VocabWord> learningAlgorithm) {
        this.learningAlgorithm = learningAlgorithm;
        getW2VFromFile();
    }

    private void getW2VFromFile() {
        for (Language language: Language.values()) {
            Word2Vec loadedVec = readModelFromFile(language);
            languageWord2VecMap.put(language,loadedVec);
        }
    }

    private void getWord2Vec() {

        t.setTokenPreProcessor(new CommonPreprocessor());

        for (Language language: languages.keySet()) {
            List<String> sentences = getSentencesFromLanguage(language);

            SentenceIterator iter = new CollectionSentenceIterator(PREPROCESSOR, sentences);
            Word2Vec vec = new Word2Vec.Builder().elementsLearningAlgorithm(learningAlgorithm)
                    .minWordFrequency(6)
                    .iterations(15)
                    .layerSize(VEC_LENGTH)
                    .seed(42)
                    .windowSize(5)
                    .iterate(iter)
                    .tokenizerFactory(t)
                    .build();

            vec.fit();
            saveModel(vec, language);
            languageWord2VecMap.put(language, vec);
        }

    }

    public  Word2Vec readModelFromFile(Language language) {
        String path = (learningAlgorithm instanceof SkipGram) ?
                language.getName() + "_model.txt" : language.getName() + "_model_" + learningAlgorithm.getCodeName() + ".txt";
        URL resource = Pan15Word2Vec.class.getClassLoader()
                .getResource("word2vec/" + path);
        try {
            return WordVectorSerializer.readWord2VecModel(Paths.get(resource.toURI()).toFile());
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void saveModel(Word2Vec model, Language language) {
        String dir = "./src/main/resources/word2vec";
        String path = (learningAlgorithm instanceof SkipGram) ?
                dir + "/" + language.getName() + "_model.txt"
                :dir + "/" + language.getName() + "_model_" + learningAlgorithm.getCodeName() + ".txt";
        WordVectorSerializer.writeWord2VecModel(model, path);
    }

    public INDArray getSentence2VecTfIdf(String sentence, Language language) {
        t.setTokenPreProcessor(new CommonPreprocessor());
        List<String> tokens = t.create(sentence).getTokens();
        double[] tfidf = new double[tokens.size()];
        List<double[]> wordEmbeddings = getWordEmbeddings(sentence, language);
        for (int i = 0; i < tfidf.length; i++) {
            tfidf[i] = Utils.tfIdf(sentence, getSentencesFromLanguage(language), tokens.get(i));
            if (wordEmbeddings.get(i) != null) {
                double[] vec = wordEmbeddings.get(i);
                for(int j = 0; j < vec.length; j++) vec[j] *= tfidf[i];
            }
        }
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for (double[] vector : wordEmbeddings) {
            if (vector != null) {
                INDArray vec = Nd4j.create(vector);
                featureVector = featureVector.add(vec);
            }
        }
        return featureVector;
    }

    public INDArray getSentence2VecBigramModel(String sentence, Language language) {
        List<double[]> wordEmbeddings = getWordEmbeddings(sentence, language);
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for (int i = 1; i < wordEmbeddings.size(); i++) {
            INDArray prev = wordEmbeddings.get(i-1) == null ? Nd4j.zeros(1, VEC_LENGTH) : Nd4j.create(wordEmbeddings.get(i-1));
            INDArray curr = wordEmbeddings.get(i) == null ? Nd4j.zeros(1, VEC_LENGTH) : Nd4j.create(wordEmbeddings.get(i));
            INDArray tanhSum = Transforms.tanh(prev.add(curr));
            featureVector = featureVector.add(tanhSum);
        }
        return featureVector;
    }
    public INDArray getSentence2VecAvg(String sentence, Language language) {
        List<double[]> wordEmbeddings = getWordEmbeddings(sentence, language);
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for (double[] vector : wordEmbeddings) {
            if (vector != null) {
                INDArray vec = Nd4j.create(vector);
                featureVector = featureVector.add(vec);
            }
        }
        featureVector.divi(wordEmbeddings.size());
        return featureVector;
    }

    public INDArray getSentence2VecSum(String sentence, Language language) {
        List<double[]> wordEmbeddings = getWordEmbeddings(sentence, language);
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for (double[] vector : wordEmbeddings) {
            if (vector != null) {
                INDArray vec = Nd4j.create(vector);
                featureVector = featureVector.add(vec);
            }
        }
        return featureVector;
    }

    public List<double[]> getWordEmbeddings(String sentence, Language language) {
        t.setTokenPreProcessor(new CommonPreprocessor());
        List<String> tokens = t.create(sentence).getTokens();
        double[] tfidf = new double[tokens.size()];
        for (int i = 0; i < tfidf.length; i++) {
            tfidf[i] = Utils.tfIdf(sentence, getSentencesFromLanguage(language), tokens.get(i));
        }
        Word2Vec loadedVec = languageWord2VecMap.get(language);
        return tokens.stream().map(loadedVec::getWordVector).collect(Collectors.toList());
    }


    public static void main(String... args) {
        new Pan15Word2Vec(new GloVe<>());

    }

    @Override
    public int getVecLength() {
        return VEC_LENGTH;
    }

    @Override
    public INDArray getVector(String sentence, Language language) {
        return getSentence2VecAvg(sentence, language);
    }
}
