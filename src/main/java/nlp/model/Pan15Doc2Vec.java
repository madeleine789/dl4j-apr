package nlp.model;

import model.Language;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;

public class Pan15Doc2Vec implements Model {
    private static HashMap<Language, HashMap<String, INDArray>> languageDoc2VecMap = loadDoc2VecsFromFile();
    private static int VEC_LENGTH = 300;
    private static HashMap<Language, HashMap<String, INDArray>> loadDoc2VecsFromFile() {
        HashMap<Language, HashMap<String, INDArray>> languageDoc2VecMap = new HashMap<>();
        for (Language language : Language.values()) {
            languageDoc2VecMap.put(language, parseLanguage(language));
        }
        return languageDoc2VecMap;
    }

    static HashMap<String, INDArray> parseLanguage(Language language) {
        HashMap<String, INDArray> doc2vecs = new HashMap<>();
        File path = new File("/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/doc2vec/" +
                language.getName() + "-doc2vec.txt");
        try {
            System.out.println(Files.readAllLines(path.toPath()).size());
            List<String> lines = Files.readAllLines(path.toPath());
            for (String l : lines) {
                String[] vec = l.split(",");
                if (vec.length == VEC_LENGTH+1) {
                    String tweet = vec[0];
                    double[] data = new double[vec.length-1];
                    for(int i = 1; i < vec.length; i++) {
                        data[i-1] = Double.parseDouble(vec[i]);
                    }
                    doc2vecs.put(tweet,Nd4j.create(data));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return doc2vecs;
    }

    public INDArray getDoc2Vec(String sentence, Language language) {
        sentence = sentence.replaceAll(",", "");
        return languageDoc2VecMap.get(language).getOrDefault(sentence, Nd4j.create(1, VEC_LENGTH));
    }

    @Override
    public int getVecLength() {
        return VEC_LENGTH;
    }

    @Override
    public INDArray getVector(String sentence, Language language) {
        if(sentence.endsWith("\"") && sentence.startsWith("\""))
            sentence = sentence.substring(1, sentence.length() - 1);
        return getDoc2Vec(sentence, language);
    }
}
