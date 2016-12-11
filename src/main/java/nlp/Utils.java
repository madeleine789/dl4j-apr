package nlp;

import java.util.List;

public class Utils {

    private static double tf(String doc, String term) {
        double result = 0;
        for (String word : doc.split(" ")) {
            if (term.equalsIgnoreCase(word))
                result++;
        }
        return result / doc.length();
    }
    private static double tf(List<String> doc, String term) {  // treat all tweets from author as one
        double result = 0;
        for (String word : doc) {
            if (term.equalsIgnoreCase(word))
                result++;
        }
        return result / doc.size();
    }

    private static double idf2(List<List<String>> docs, String term) {
        double n = 0;
        for (List<String> doc : docs) {
            for (String word : doc) {
                if (term.equalsIgnoreCase(word)) {
                    n++;
                    break;
                }
            }
        }
        return Math.log(docs.size() / n);
    }

    private static double idf(List<String> docs, String term) {
        double n = 0;
        for (String doc : docs) {
            for (String word : doc.split(" ")) {
                if (term.equalsIgnoreCase(word)) {
                    n++;
                    break;
                }
            }
        }
        return Math.log(docs.size() / n);
    }

    public static double tfIdf(List<String> doc, List<List<String>> docs, String term) {
        return tf(doc, term) * idf2(docs, term);
    }

    public static double tfIdf(String doc, List<String> docs, String term) {
        return tf(doc, term) * idf(docs, term);
    }

    //average of (word vectors multiplied by tfidf)
}
