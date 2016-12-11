package parsing.pan15;

import nlp.Pan15SentencePreProcessor;
import nlp.Pan15Word2Vec;
import org.junit.Test;
import model.Author;
import model.Language;

import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class PreProcessorTest {

    @Test
    public void shouldTest() {
        Pan15Parser parser = new Pan15Parser();
        Pan15SentencePreProcessor preProcessor = new Pan15SentencePreProcessor();
        HashMap<String, Pan15Author> english = parser.parseLanguage(Language.ENGLISH);
        List<String> englishSentences = english.values().stream().map(Author::getDocuments)
                .collect(Collectors.toList())
                .stream().flatMap(List::stream).collect(Collectors.toList()).stream()
                .map(preProcessor::preProcess).collect(Collectors.toList());

        List<double[]> wordEmbeddings = new Pan15Word2Vec().getWordEmbeddings(englishSentences.get(10), Language.ENGLISH);
        wordEmbeddings.stream().filter(Objects::nonNull).forEach(w -> System.out.println(w.length));

    }
}
