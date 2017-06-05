package nlp.model;

import model.Language;
import nlp.Pan15SentencePreProcessor;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

import static nlp.Utils.getSentencesFromLanguage;
import static nlp.Utils.normalize;

public class Pan15BagOfWords implements Model {

    private static HashMap<Language, LinkedHashMap<String, Integer>> bows = getBOWs();
    private final static int VEC_LENGTH = 5000;
    private boolean bigramBow = true;

    public Pan15BagOfWords(boolean bigramBow) {
        this.bigramBow = bigramBow;
    }


    private static HashMap<Language, LinkedHashMap<String, Integer>> getBOWs() {
        HashMap<Language, LinkedHashMap<String, Integer>> bows = new HashMap<>();
        for (Language language: Language.values()) {
            bows.put(language,getBagOfWordsWithCounts(language));
        }
        return bows;
    }

    private static LinkedHashMap<String, Integer> getBagOfWordsWithCounts(Language language) {
        HashMap<String, Integer> bagOfWords = new HashMap<>();
        List<String> sentences = getSentencesFromLanguage(language);
        SentenceIterator iter = new CollectionSentenceIterator(new Pan15SentencePreProcessor(), sentences);
        while(iter.hasNext()) {
            String sentence = iter.nextSentence();
            for(String word : sentence.split("\\s+")) {
                word =  normalize(word);
                if (Objects.equals(word, "") || (word.length() == 1 && word.matches("\\p{Punct}"))) continue;
                bagOfWords.put(word, bagOfWords.getOrDefault(word, 0) + 1);
            }
        }
        LinkedHashMap<String, Integer> sorted = new LinkedHashMap<>();
        final int[] count = {0};
        bagOfWords.entrySet().stream()
                .sorted(Map.Entry.comparingByValue(Collections.reverseOrder())).forEach(
                entry -> {
                    if (count[0] < VEC_LENGTH) sorted.put(entry.getKey(), entry.getValue());
                    count[0]++;
                }
        );
        return sorted;
    }

    public static LinkedList<String> getBagOfWords(Language language) {
        LinkedHashMap<String, Integer> bow = bows.get(language);
        return new LinkedList<>(bow.keySet());
    }

    public INDArray getBinaryBoWVector(String sentence, Language language) {
        LinkedList<String> keys = getBagOfWords(language);
        SentenceIterator iter = new CollectionSentenceIterator(new Pan15SentencePreProcessor(), Collections.singletonList(sentence));
        sentence = iter.nextSentence();
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for(String word : sentence.split("\\s+")) {
            word =  normalize(word);
            int col = keys.indexOf(word);
            if (col > -1) featureVector.putScalar(0, col, 1);
        }
        return featureVector;
    }

    public INDArray getBoWVector(String sentence, Language language) {
        LinkedList<String> keys = getBagOfWords(language);
        SentenceIterator iter = new CollectionSentenceIterator(new Pan15SentencePreProcessor(), Collections.singletonList(sentence));
        sentence = iter.nextSentence();
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for(String word : sentence.split("\\s+")) {
            word =  normalize(word);
            int col = keys.indexOf(word);
            if (col > -1)
                featureVector.putScalar(0, col, featureVector.getColumn(col).getInt(0) + 1);
        }
        featureVector.divi(VEC_LENGTH);
        return featureVector;
    }

    private List<INDArray> getWordEmbeddings(String sentence, Language language) {
        LinkedList<String> keys = getBagOfWords(language);
        List<INDArray> embeddings = new LinkedList<>();
        for(String word : sentence.split("\\s+")) {
            INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
            word =  normalize(word);
            int col = keys.indexOf(word);
            if (col > -1) featureVector.putScalar(0, col, featureVector.getColumn(col).getFloat(0) + 1);
            featureVector.divi(VEC_LENGTH);
            embeddings.add(featureVector);
        }
        return embeddings;
    }

    public INDArray getBoWBigramModel(String sentence, Language language) {
        List<INDArray> wordEmbeddings = getWordEmbeddings(sentence, language);
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for (int i = 1; i < wordEmbeddings.size(); i++) {
            INDArray prev = wordEmbeddings.get(i-1) == null ? Nd4j.zeros(1, VEC_LENGTH) : wordEmbeddings.get(i-1);
            INDArray curr = wordEmbeddings.get(i) == null ? Nd4j.zeros(1, VEC_LENGTH) : wordEmbeddings.get(i);
            INDArray tanhSum = Transforms.tanh(prev.add(curr));
            featureVector = featureVector.add(tanhSum);
        }
        return featureVector;
    }

    public List<INDArray> parseLanguage(Language language) {
        List<INDArray> bows = new ArrayList<>();
        File path = new File("./src/main/resources/pan14/" +
                language.getName() + ".csv");
        try {
            List<String> lines = Files.readAllLines(path.toPath());
            lines = lines.subList(1, lines.size());
            for (String l : lines) {
                String[] vec = l.split(",");
                String tweet = vec[0];
                INDArray bow = getVector(tweet, language);
                bows.add(bow);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bows;
    }

    @Override
    public int getVecLength() {
        return VEC_LENGTH;
    }

    @Override
    public INDArray getVector(String sentence, Language language) {
        return (bigramBow) ? getBoWBigramModel(sentence, language): getBoWVector(sentence, language);
    }
}
