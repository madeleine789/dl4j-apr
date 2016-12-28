package nlp;

import model.Author;
import model.Language;
import parsing.CorpusParser;
import parsing.pan15.Pan15Author;
import parsing.pan15.Pan15Parser;

import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

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

    private static CorpusParser<Pan15Author> parser = new Pan15Parser();
    private static HashMap<Language, HashMap<String, Pan15Author>> languages = parser.parseCSVCorpus();
    private static HashMap<Language, List<String>>  sentences = new HashMap<>();

    public static List<String> getSentencesFromLanguage(Language language) {
        if (!sentences.containsKey(language)) {
            List<String> s = languages.get(language).values().stream().map(Author::getDocuments)
                    .collect(Collectors.toList())
                    .stream().flatMap(List::stream).collect(Collectors.toList());
            sentences.put(language, s);
        }
        return sentences.get(language);
    }

    public static HashMap<Language, HashMap<String, Pan15Author>> getLanguages() {
        return languages;
    }

    public static String normalize(String word) {
        word =  (word.startsWith("P_")) ? word : word.toLowerCase();
        word = word.replaceAll("[^A-Za-z]$", "");
        word = word.replaceAll("^[^A-Za-z]$", "");
        word = word.replaceAll("^[^A-Za-z@#]", "");
        return word;
    }

}
