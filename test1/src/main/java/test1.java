import javafx.util.Pair;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.util.*;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class test1 {

    private static final Logger log = LoggerFactory.getLogger(test1.class);
    private static final int seed = 1234;
    private static final int nChannel = 1;
    private static final int nClasses = 3;
    private static final int nEpochs = 20;
    private static final int height = 48;
    private static final int width = 48;
    private static final int batchSize = 200;
    private static final Random rng = new Random(13);
    private static List<String> labels;
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-model/test2model.zip");
    private static MultiLayerNetwork model;


    public static void main(String[] args) throws Exception {

        //STEP 0 : Load trained model from previous execution

        if (modelFilename.exists()) {
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
            model = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
            testImage();
        } else {

        //Step 1: Load image data

            File trainDir = new ClassPathResource("faces/trn").getFile();
            File testDir = new ClassPathResource("faces/tst").getFile();
            FileSplit trainSplit = new FileSplit(trainDir, allowedExtensions, rng);
            FileSplit testSplit = new FileSplit(testDir, allowedExtensions, rng);

            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

            ImageRecordReader trainRR = new ImageRecordReader(height, width, nChannel, labelMaker);
            ImageRecordReader testRR = new ImageRecordReader(height, width, nChannel, labelMaker);

            trainRR.initialize(trainSplit);
            testRR.initialize(testSplit);

            int numLabels = trainRR.numLabels();

            int labelIndex = 1;

            RecordReaderDataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, labelIndex, numLabels);
            RecordReaderDataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, labelIndex, numLabels);

            int batchIndex = 0;
            while (trainIter.hasNext()) {
            DataSet ds = trainIter.next();

            batchIndex += 1;
            System.out.println("\nBatch number: " + batchIndex);
            System.out.println("Feature vector shape: " + Arrays.toString(ds.getFeatures().shape()));
            System.out.println("Label vector shape: " + Arrays.toString(ds.getLabels().shape()));
            }

            //Print labels
            labels = trainIter.getLabels();
            System.out.println(Arrays.toString(labels.toArray()));

            trainIter.setPreProcessor(scaler);
            testIter.setPreProcessor(scaler);


        //Step 2: Create CNN/transfer learning

            Map<Integer, Double> schedule = new HashMap<>();

            schedule.put(0, 0.01);
            schedule.put(14, 0.001);


            MapSchedule learningRateSchedule = new MapSchedule(ScheduleType.EPOCH, schedule);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .weightInit(WeightInit.XAVIER)
                    .l2(0.0001)
                    .updater(new Nesterovs(learningRateSchedule))
                    .list()
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(nChannel)
                            .stride(1, 1)
                            .activation(Activation.IDENTITY)
                            .nOut(250)
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(3, 3)
                            .stride(2, 2)
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(5, 5)
                            .stride(1, 1)
                            .activation(Activation.IDENTITY)
                            .nOut(200)
                            .build())
                    .layer(3, new ConvolutionLayer.Builder(5, 5)
                            .stride(1, 1)
                            .activation(Activation.RELU)
                            .nOut(150)
                            .build())
                    .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(3, 3)
                            .stride(2, 2)
                            .build())
                    .layer(5, new DenseLayer.Builder()
                            .activation(Activation.TANH)
                            .nOut(100)
                            .build())
                    .layer(6, new DenseLayer.Builder()
                            .activation(Activation.TANH)
                            .nOut(50)
                            .build())
                    .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX)
                            .nOut(nClasses)
                            .build())
                    .setInputType(InputType.convolutionalFlat(height, width, 1))
                    .build();


            model = new MultiLayerNetwork(conf);
            model.init();


            StatsStorage storage = new InMemoryStatsStorage();
            UIServer server = UIServer.getInstance();
            server.attach(storage);
            model.setListeners(new StatsListener(storage));

            log.info("Train model");

        //Step 3: Evaluate

            for (int i = 0; i < nEpochs; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                log.info("Completed epoch " + i);
            }

            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");

            Evaluation eval1 = model.evaluate(trainIter);
            System.out.println(eval1.stats());
            Evaluation eval = model.evaluate(testIter);
            System.out.println(eval.stats());


            log.info("Program End");


        }
    }

    private static void testImage() throws Exception {
        String testImagePATH = "D:\\Desktop\\facial emotion\\m.jpg";    //change path to own image dir
        File file = new File(testImagePATH);
        System.out.println(String.format("You are using this image file located at %s", testImagePATH));
        NativeImageLoader nil = new NativeImageLoader(48, 48, 1);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);


        INDArray image = nil.asMatrix(file);
        scaler.transform(image);


        Mat opencvMat = imread(testImagePATH);
        INDArray outputs = model.output(image);
        INDArray op = Nd4j.argMax(outputs, 1);

        int ans = op.getInt(0);

        if (ans == 0) {
            log.info("Emotion : Angry");
        }
        if (ans == 1) {
            log.info("Emotion : Happy");
        }
        if (ans == 2) {
            log.info("Emotion : Neutral");
        }

        log.info("Label:         " + Nd4j.argMax(outputs, 1));
        log.info("Probabilities: " + outputs.toString());

        imshow("Input Image", opencvMat);

        if (waitKey(0) == 27) {
            destroyAllWindows();
        }

    }

}

