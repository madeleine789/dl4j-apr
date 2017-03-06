package nlp;

import model.Language;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;

public class Pan15Tweet2Vec {
    private static HashMap<Language, HashMap<String, INDArray>> languageTweet2VecMap = loadTweet2VecsFromFile();
    public final static int VEC_LENGTH = 500;
    private static HashMap<Language, HashMap<String, INDArray>> loadTweet2VecsFromFile() {
        HashMap<Language, HashMap<String, INDArray>> languageTweet2VecMap = new HashMap<>();
        for (Language language : Language.values()) {
            languageTweet2VecMap.put(language, parseLanguage(language));
        }
        return languageTweet2VecMap;
    }

    static HashMap<String, INDArray> parseLanguage(Language language) {
        HashMap<String, INDArray> tweet2vecs = new HashMap<>();
        File path = new File("/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/tweet2vec/" +
                language.getName() + "-tweet2vec.txt");
        try {
            List<String> lines = Files.readAllLines(path.toPath());
            for(String l: lines) {
                String[] vec = l.split(",");
                if (vec.length == VEC_LENGTH+1) {
                    String tweet = vec[0];
                    double[] data = new double[vec.length-1];
                    for(int i = 1; i < vec.length; i++) {
                        data[i-1] = Double.parseDouble(vec[i]);
                    }
                    INDArray normalized = Transforms.normalizeZeroMeanAndUnitVariance(Nd4j.create(data));
                    tweet2vecs.put(tweet, normalized);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tweet2vecs;
    }

    public INDArray getTweet2Vec(String sentence, Language language) {
        sentence = sentence.replaceAll(",", "");
        return languageTweet2VecMap.get(language).getOrDefault(sentence, Nd4j.create(1, VEC_LENGTH));
    }
}
