package model;

public enum Personality {
    O(6), C(5), E(2), A(4), N(3);

    private int index;

    Personality(int index) {
        this.index = index;
    }
     public int getIndex() {
        return this.index;
     }
}
