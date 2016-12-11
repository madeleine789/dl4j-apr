package parsing.pan15;

import nlp.Pan15SentencePreProcessor;
import nlp.Pan15Word2Vec;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Assert;
import org.junit.Test;
import model.Author;
import model.Language;

import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class Pan15Word2VecTest {

    @Test
    public void shouldLoadAndCreateSameWord2Vec() {
        //given
        Pan15Parser parser = new Pan15Parser();
        HashMap<String, Pan15Author> english = parser.parseCSVCorpus().get(Language.ENGLISH);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        List<String> englishSentences = english.values().stream().map(Author::getDocuments)
                .collect(Collectors.toList())
                .stream().flatMap(List::stream).collect(Collectors.toList());

        SentenceIterator englishIter = new CollectionSentenceIterator(new Pan15SentencePreProcessor(), englishSentences);
        // when
        Word2Vec englishVec = new Word2Vec.Builder()
                .minWordFrequency(6)
                .iterations(15)
                .layerSize(250)
                .seed(42)
                .windowSize(5)
                .iterate(englishIter)
                .tokenizerFactory(t)
                .build();

        englishVec.fit();
        Word2Vec loadedEnglishVec = Pan15Word2Vec.readModelFromFile(Language.ENGLISH);
        loadedEnglishVec.setTokenizerFactory(t);
        loadedEnglishVec.setSentenceIter(englishIter);

        //then
        Assert.assertNotNull(loadedEnglishVec);
        System.out.println(englishVec.wordsNearest("death", 10));
        System.out.println(loadedEnglishVec.wordsNearest("life", 10));
    }
}
