package nn.dl4j;

import lombok.Getter;
import lombok.Setter;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public abstract class AbstractDataSetIterator implements DataSetIterator {
    protected RecordReader recordReader;
    protected int batchSize = 10;
    protected int maxNumBatches = -1;
    protected int batchNum = 0;
    protected Iterator<List<Writable>> sequenceIter;
    protected DataSet last;
    protected boolean useCurrent = false;
    private DataSetPreProcessor dataSetPreProcessor;

    public AbstractDataSetIterator() {}
    public AbstractDataSetIterator(RecordReader recordReader, int batchSize, int maxNumBatches) {
        this.batchSize = batchSize;
        this.maxNumBatches = maxNumBatches;
        this.recordReader = recordReader;
    }

    @Getter
    @Setter
    private boolean collectMetaData = false;

    @Override
    public DataSet next(int num) {
        if (useCurrent) {
            useCurrent = false;
            if (dataSetPreProcessor != null) dataSetPreProcessor.preProcess(last);
            return last;
        }

        List<DataSet> dataSets = new ArrayList<>();
        List<RecordMetaData> meta = (collectMetaData ? new ArrayList<RecordMetaData>() : null);
        for (int i = 0; i < num; i++) {
            if (!hasNext())
                break;
            if (recordReader instanceof SequenceRecordReader) {
                if (sequenceIter == null || !sequenceIter.hasNext()) {
                    List<List<Writable>> sequenceRecord = ((SequenceRecordReader) recordReader).sequenceRecord();
                    sequenceIter = sequenceRecord.iterator();
                }

                List<Writable> record = sequenceIter.next();
                dataSets.add(getDataSet(record));
            } else {
                if(collectMetaData){
                    Record record = recordReader.nextRecord();
                    dataSets.add(getDataSet(record.getRecord()));
                    meta.add(record.getMetaData());
                } else {
                    List<Writable> record = recordReader.next();
                    dataSets.add(getDataSet(record));
                }
            }
        }
        batchNum++;

        if(dataSets.isEmpty())
            return new DataSet();

        DataSet ret = DataSet.merge(dataSets);
        if(collectMetaData){
            ret.setExampleMetaData(meta);
        }
        last = ret;
        if (dataSetPreProcessor != null) dataSetPreProcessor.preProcess(ret);
        //Add label name values to dataset
        if (recordReader.getLabels() != null) ret.setLabelNames(recordReader.getLabels());
        return ret;
    }

    abstract DataSet getDataSet(List<Writable> record);

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        } else
            return last.numInputs();

    }

    @Override
    public int totalOutcomes() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numOutcomes();
        } else
            return last.numOutcomes();


    }

    @Override
    public boolean resetSupported(){
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        batchNum = 0;
        recordReader.reset();
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return (recordReader.hasNext() && (maxNumBatches < 0 || batchNum < maxNumBatches));
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return recordReader.getLabels();
    }

}
