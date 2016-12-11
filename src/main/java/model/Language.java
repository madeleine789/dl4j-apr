package model;

public enum Language {
    ENGLISH, DUTCH, ITALIAN, SPANISH;

    public String getName() {
        return this.name().toLowerCase();
    }
}
