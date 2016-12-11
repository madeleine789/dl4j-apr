package nlp;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import model.Author;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import parsing.CorpusParser;
import model.Language;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import parsing.pan15.Pan15Author;
import parsing.pan15.Pan15Parser;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Pan15Word2Vec implements Word2VecBuilder<Pan15Parser> {

    public static final Pan15SentencePreProcessor PREPROCESSOR = new Pan15SentencePreProcessor();
    private static HashMap<Language, Word2Vec> languageWord2VecMap = new HashMap<>();
    private TokenizerFactory t = new DefaultTokenizerFactory();
    private CorpusParser<Pan15Author> parser = new Pan15Parser();
    HashMap<Language, HashMap<String, Pan15Author>> languages = parser.parseCSVCorpus();
    private static HashMap<Language, List<String>>  sentences = new HashMap<>();
    public static final Integer VEC_SIZE = 250;

    public Pan15Word2Vec() {
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
            Word2Vec vec = new Word2Vec.Builder()
                    .minWordFrequency(6)
                    .iterations(15)
                    .layerSize(VEC_SIZE)
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

    public static Word2Vec readModelFromFile(Language language) {
        URL resource = Pan15Word2Vec.class.getClassLoader()
                .getResource("word2vec/" + language.getName() + "_model.txt");
        try {
            return WordVectorSerializer.readWord2VecModel(Paths.get(resource.toURI()).toFile());
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void saveModel(Word2Vec model, Language language) {
        String dir = "./src/main/resources/word2vec";
        WordVectorSerializer.writeWord2VecModel(model, dir + "/" + language.getName() + "_model.txt");
    }

    public INDArray getSentence2Vec(String sentence, Language language) {
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
        INDArray featureVector = Nd4j.zeros(1, VEC_SIZE);
        for (double[] vector : wordEmbeddings) {
            if (vector != null) {
                INDArray vec = Nd4j.create(vector);
                featureVector.addi(vec);
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

        private List<String> getSentencesFromLanguage(Language language) {
        if (!sentences.containsKey(language)) {
            List<String> s = languages.get(language).values().stream().map(Author::getDocuments)
                    .collect(Collectors.toList())
                    .stream().flatMap(List::stream).collect(Collectors.toList());
            sentences.put(language, s);
        }
        return sentences.get(language);
    }

}
