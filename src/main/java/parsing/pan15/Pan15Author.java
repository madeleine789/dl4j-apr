package parsing.pan15;

import model.Personality;
import model.Author;

import javax.xml.bind.annotation.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


@XmlRootElement(name = "author")
@XmlAccessorType(XmlAccessType.FIELD)
public class Pan15Author implements Author {
    @XmlAttribute private String id;
    @XmlAttribute private String lang;
    @XmlElement(name = "document") private List<String> documents;
    private String gender;
    private String age;
    private HashMap<Personality, Double> personality = new HashMap<>();

    public String getId() {
        return id;
    }

    public String getLanguage() {
        return lang;
    }

    public List<String> getDocuments() {
        return documents;
    }

    public void setId(String id) {
        this.id = id;
    }

    public void setLang(String lang) {
        this.lang = lang;
    }

    public void setDocuments(List<String> documents) {
        this.documents = documents;
    }

    public void addDocument(String doc) {
        if (documents!= null && !documents.isEmpty()) documents.add(doc);
        else {
            documents = new ArrayList<>();
            documents.add(doc);
        }
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public void setAge(String age) {
        this.age = age;
    }

    public void setPersonality(Personality trait, double value) {
        personality.put(trait, value);
    }

    public HashMap<Personality, Double> getPersonality() {
        return personality;
    }
}
