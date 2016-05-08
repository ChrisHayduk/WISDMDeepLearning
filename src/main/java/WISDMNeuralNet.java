import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

/**
 * Created by chris on 1/31/16.
 */
public class WISDMNeuralNet {
    private static int iterations = 5;
    private static int num_people = 18; //Change back to 18
    private static int width = 100;
    private static int height = 6;
    private static int numChannels = 1;
    private static int batchSize = 5;
    private static RecordReaderDataSetIterator iter;
    private static ImageRecordReader recordReader;
    private static  String labeledPath = System.getProperty("user.dir")+"/training/";
    private static List<String> labels = new ArrayList<>();
    private static int nEpochs = 5;

    public static void main(String[] args) throws Exception {
        loadData();
        System.out.println("Configuring model...");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(iterations)
                .regularization(true).l2(.0005) //Arbitrary
                .learningRate(0.001)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(numChannels).nOut(20).activation("identity").padding(2,2).stride(1,1).dropOut(0.5).build()
                )
                .layer(1,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                        .kernelSize(2,2).stride(1,1).build()
                )
                .layer(2, new ConvolutionLayer.Builder()
                        .nIn(numChannels).nOut(50).activation("identity").padding(2,2).stride(1,1).dropOut(0.5).build()
                )
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                        .kernelSize(2,2).stride(1,1).build()
                )
                .layer(4, new DenseLayer.Builder().activation("relu").nOut(50).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nOut(num_people).activation("softmax").build())
                .backprop(true).pretrain(false)
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(height, width, numChannels));
        new ConvolutionLayerSetup(builder,height,width,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //model.setListeners(new HistogramIterationListener(iterations));

        System.out.println("Training model....");
        int counter = 0;

        for(int i = 0; i < nEpochs; i++){
            model.fit(iter);
            System.out.println("\n*** Completed epoch {} ***");
            System.out.println(i);

            System.out.println("\nEvaluate model...");
            Evaluation eval = new Evaluation(num_people);

            while(iter.hasNext()){
                DataSet next = iter.next();
                INDArray output = model.output(next.getFeatureMatrix(), false);
                eval.eval(next.getLabels(), output);
            }

            System.out.println("\n" + eval.stats());
            iter.reset();
        }

        System.out.println("******Program finished******");
        /*
        while(iter.hasNext()){
            counter++;
            DataSet next = iter.next();
            System.out.println(Arrays.toString(next.getFeatureMatrix().shape()));
            System.out.println(Arrays.toString(next.getLabels().shape()));
            System.out.println(next.labelCounts());
            System.out.println("counter: " + counter + '\n');
            model.fit(next);
        }
        iter.reset();
        counter = 0;
        //Testing on the input set because IDGAF while I'm working out the bugs
        //Later, use an actual test set
        System.out.println("Testing model...");
        Evaluation eval = new Evaluation();

        while(iter.hasNext()){
            counter++;
            System.out.println("counter: " + counter);
            DataSet next = iter.next();
            INDArray output = model.output(next.getFeatureMatrix(), false);
            eval.eval(next.getLabels(), output);
        }

        System.out.println(eval.stats());
*/

    }


    public static void loadData() throws Exception{
    /*    for (File file : new File(labeledPath).listFiles()){
            labels.add(file.getName());
            System.out.println(file.getName());
        }


        System.out.println(labels.size());

    */

       // recordReader = new ImageRecordReader(height,width,true, labels);
        recordReader = new ImageRecordReader(width, height, numChannels, true);

        recordReader.initialize(new FileSplit(new File(labeledPath)));

        //iter = new RecordReaderDataSetIterator(recordReader, 6*100, labels.size());
        iter = new RecordReaderDataSetIterator(recordReader, batchSize, -1, 18);

    }
}
