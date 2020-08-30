package neuralNet;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import Jama.Matrix;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.Vector.Norm;
/**
 * A feed-forward fully connected deep neural network
 * @author Zachary
 */
public class NeuralNet {
	private WeightLayer[] weights;
	private int ins;
	private int outs;
	private int[] layers;
	private int lossLast = 0;
	private int lossTotal = 0;
	private int batchSize = 0;
	private int batchCount = 0;
	protected NeuralNet(int ins, int outs, int[] layers, WeightLayerSerializable[] weights)
	{
		this(ins,layers,outs,0,0,new Random());
		for(int i=0;i<weights.length;i++)
		{
			this.weights[i] = weights[i].toWeightLayer();
		}
	}
	protected WeightLayerSerializable[] getWeightSerializable()
	{
		WeightLayerSerializable[] ret = new WeightLayerSerializable[weights.length];
		for(int i=0;i<ret.length;i++)
		{
			ret[i] = new WeightLayerSerializable(weights[i]);
		}
		return ret;
	}
	/**
	 * Sets the mini-batch size for training
	 * Training with larger batches results in slower training
	 */
	public void setTrainingBatchSize(int n)
	{
		batchSize = n;
	}
	/**
	 * Returns a copy of the list representing the nodes per hidden layer
	 * @return
	 */
	public int[] getInternalLayers()
	{
		return layers.clone();
	}
	/**
	 * Returns the input dimension size
	 */
	public int getIns()
	{
		return ins;
	}
	/**
	 * Returns the output dimension size
	 */
	public int getouts()
	{
		return outs;
	}
	void trackLoss()
	{
		lossTotal = 0;
	}
	public double lossLast()
	{
		return lossLast;
	}
	double getLossAndReset()
	{
		lossLast = lossTotal;
		lossTotal = 0;
		return lossLast;
	}
	/**
	 * Returns the distribution of a 1-hot embedding according to all input dimensions
	 * @param alternativeSigmoid whether to use RELU or not
	 * @param layerNumber the internal layer to sample
	 * @return an array containing a hidden layer activation for each input dimension in a 1-hot fashion
	 */
	public nVector[] samplesWeightLayerOneHot(boolean alternativeSigmoid, int layerNumber)
	{
		nVector[] retval = new nVector[weights[0].ins];
		for(int i = 0; i < retval.length; i++)
		{
			nVector inputs = new nVector(weights[0].ins, i);
			for(int ln = 0; ln<=layerNumber; ln++)
			{
				inputs=weights[ln].applyWithBias(inputs);
				if(alternativeSigmoid)
				{
					inputs=inputs.altSigmoidize();
				} else
					inputs=inputs.sigmoidize();
			}
			retval[i] = inputs;
		}
		return retval;
	}
	/**
	 * Returns the distribution of a the input vectors once embedded
	 * @param alternativeSigmoid whether to use RELU or not
	 * @param layerNumber the internal layer to sample
	 * @return an array containing a hidden layer activation for each input vector
	 */
	public nVector[] samplesWeightLayer(nVector[] ins, boolean alternativeSigmoid, int layerNumber)
	{
		nVector[] retval = new nVector[ins.length];
		for(int i = 0; i < ins.length; i++)
		{
			nVector inputs = ins[i];
			for(int ln = 0; ln<=layerNumber; ln++)
			{
				inputs=weights[ln].applyWithBias(inputs);
				if(alternativeSigmoid)
				{
					inputs=inputs.altSigmoidize();
				} else
					inputs=inputs.sigmoidize();
			}
			retval[i] = inputs;
		}
		return retval;
	}
	/**
	 * Creates a matrix that produces input vectors for the given hidden layer, 0 indexed via PCA
	 * @return The matrix V such that weights[layer] = UΣV*
	 */
	public WeightLayer principalComponentAnalysis(int layer)
	{
		WeightLayer focus = weights[layer];
		DenseMatrix m = focus.x;
		double[] matrixData = m.getData();
		Matrix decomp = new Matrix(matrixData, m.numRows());
		Matrix v = decomp.svd().getV();
		DenseMatrix vfinal = new DenseMatrix(v.getArrayCopy());
		WeightLayer ret = new WeightLayer(vfinal,new DenseVector(vfinal.numRows()));
		return ret;
	}
	
	/**
	 * Returns the eigenvalues of the given layer, 0 indexed, sorted
	 * @return The matrix V such that weights[layer] = UΣV*
	 */
	public nVector pcaEigenvalues(int layer)
	{
		WeightLayer focus = weights[layer];
		DenseMatrix m = focus.x;
		double[] matrixData = m.getData();
		Matrix decomp = new Matrix(matrixData, m.numRows());
		double[] d = decomp.svd().getSingularValues();
		return new nVector(d);
	}
	/**
	 * Returns an autoencoder which does not update when this NeuralNet updates
	 * @param layerNumber The layer to use as the latent (compressed) space
	 * @return a detached autoencoder
	 */
	public Encoder getImmutableAutoEncoder(int layerNumber)
	{
		//we have in, l0, l1, ... ln, out
		//encoder is in, l0, ... lnumber-1, lnumber
		//decoder is lnumber, lnumber+1, ... ln, out
		final int subLen = layers.length-layerNumber-1;
		NeuralNet encoder = new NeuralNet(ins,new int[layerNumber],layers[layerNumber], 0, 0, new Random());
		NeuralNet decoder = new NeuralNet(layers[layerNumber],new int[subLen],outs, 0, 0, new Random());
		for(int i=0;i<=layers.length;i++)
		{
			if(i<=layerNumber)
			{
				encoder.weights[i]=weights[i].clone();
			} else {
				decoder.weights[i-layerNumber-1]=weights[i].clone();
			}
		}
		return new Encoder(encoder,decoder);
	}
	/**
	 * Returns an autoencoder which does update in sync with when this NeuralNet updates
	 * @param layerNumber The layer to use as the latent (compressed) space
	 * @return an attached autoencoder
	 */
	public Encoder getMutableAutoEncoder(int layerNumber)
	{
		//we have in, l0, l1, ... ln, out
		//encoder is in, l0, ... lnumber-1, lnumber
		//decoder is lnumber, lnumber+1, ... ln, out
		final int subLen = layers.length-layerNumber-1;
		NeuralNet encoder = new NeuralNet(ins,new int[layerNumber],layers[layerNumber], 0, 0, new Random());
		NeuralNet decoder = new NeuralNet(layers[layerNumber],new int[subLen],outs, 0, 0, new Random());
		for(int i=0;i<=layers.length;i++)
		{
			if(i<=layerNumber)
			{
				encoder.weights[i]=weights[i];
			} else {
				decoder.weights[i-layerNumber-1]=weights[i];
			}
		}
		return new Encoder(encoder,decoder);
	}
	
	public Encoder getGAN(int layerNumber)
	{
		final int subLen = layers.length-layerNumber-1;
		NeuralNet encoder = new NeuralNet(ins,new int[layerNumber],layers[layerNumber], 0, 0, new Random());
		NeuralNet decoder = new NeuralNet(layers[layerNumber],new int[subLen],outs, 0, 0, new Random());
		for(int i=0;i<=layers.length;i++)
		{
			if(i<=layerNumber)
			{
				encoder.weights[i]=weights[i];
			} else {
				decoder.weights[i-layerNumber-1]=weights[i];
			}
		}
		return new Encoder(this,encoder,decoder,layerNumber+1,layers.length-layerNumber-1);
	}
	/**
	 * Writes this NeuralNet to the given file
	 * @param f The file to save this neural net in
	 * @throws IOException when loading the file fails or NANs are present in the net
	 */
	public void writeToFile(File f) throws IOException
	{
		for(WeightLayer d:weights)
		{
			if(!d.safe())
				throw new IOException("cannot save NANs to file");
		}
		f.createNewFile();
		FileOutputStream file = new FileOutputStream(f); 
		ObjectOutputStream writer = new ObjectOutputStream(file); 
		writer.writeObject(new NeuralNetSerializable(this)); 
		writer.close(); 
		file.close(); 
	}
	/**
	 * Constructs a NeuralNet from the given file in the same format as written by writeToFile
	 * @param f the file to load the neural net from
	 * @throws IOException if the file could not be found or could not be read
	 * @throws ClassNotFoundException the class does not exist???
	 */
	public static NeuralNet fromFile(File f) throws IOException, ClassNotFoundException
	{
		FileInputStream file = new FileInputStream(f);
		ObjectInputStream reader = new ObjectInputStream(file);
		NeuralNetSerializable sr = (NeuralNetSerializable)reader.readObject();
		return sr.toNeuralNet();
	}
	
	/**
	 * Creates a disconnected (deep) copy of this NeuralNet
	 * @return
	 */
	public NeuralNet copy()
	{
		return randomAlter(0);
	}
	/**
	 * Constructs a NeuralNet with num of its weights changed
	 * @param num the number of weights to change
	 * @return
	 */
	NeuralNet randomAlter(int num)
	{
		NeuralNet neww = new NeuralNet(ins,layers,outs, 0, 0, new Random());
		for(int i=0; i<weights.length;i++)
		{
			neww.weights[i]=weights[i].clone();
		}
		Random rng = new Random();
		int maxL=weights.length;
		for(int i=0;i<num;i++)
		{
			WeightLayer layer = neww.weights[rng.nextInt(maxL)];
			DenseMatrix l = layer.x;
			l.add(rng.nextInt(layer.outs), rng.nextInt(layer.ins), rng.nextDouble()*2-1.0);
		}
		return neww;
	}
	
	/**
	 * Constructs a deep NeuralNet with the given configuration where each weight is given a random value according to a gaussian distribution
	 * @param ins dimensionality of the input vectors
	 * @param layers an array containing the dimensionality of each hidden layer
	 * @param outputs dimensionality of the output vectors
	 * @param sigma the standard deviation of the random initialization
	 * @param mean the mean of the random initialization. This should be 0 in most contexts
	 * @param rng the random number generator
	 */
	public NeuralNet(int ins, int[] layers, int outputs, double sigma, double mean, Random rng)
	{
		this.ins=ins;
		this.layers=layers;
		this.outs=outputs;
		weights = new WeightLayer[layers.length+1];
		
		if(layers.length==0)
		{
			weights[0]=new WeightLayer(ins,outputs, sigma, mean, rng);
		} else {
			weights[0]=new WeightLayer(ins,layers[0], sigma, mean, rng);
			for(int i=1;i<layers.length;i++)
			{
				weights[i]=new WeightLayer(layers[i-1],layers[i], sigma, mean, rng);
			}
			weights[layers.length]=new WeightLayer(layers[layers.length-1],outputs, sigma, mean, rng);
		}
	}
	/**
	 * Runs the neural network forward on an input vector without training
	 * @param inputs the input vector
	 * @param sigmoid whether or not to apply the sigmoid function to the output
	 * @param altSigmoid whether to use leaky RELU instead of sigmoid
	 * @return the result of the neural net's calculation
	 */
	public nVector calculate(nVector inputs, boolean sigmoid, boolean altSigmoid)
	{
		nVector expected=new nVector(weights[weights.length-1].outs); 
		return train(inputs,expected, 0, sigmoid, altSigmoid?1:0);
	}
	
	/**
	 * Trains the neural network on an input and output vector
	 * @param inputs the input vector
	 * @param expected the output vector the neural net should try to fit
	 * @param epsilon the learning rate
	 * @param sigmoid whether or not to apply the sigmoid function to the output 
	 * @param alternativeSigmoid whether to use leaky RELU instead of sigmoid
	 * @return the result of the neural net's calculation
	 */
	public nVector train(nVector inputs, nVector expected, double epsilon, boolean sigmoid, double alternativeSigmoid)
	{
		return trainSpecific(inputs, expected, epsilon, sigmoid, alternativeSigmoid, 0, weights.length);
	}

	protected nVector trainSpecific(nVector inputs, nVector expected, double epsilon, boolean sigmoid, double alternativeSigmoid, int layersBegin, int layersEnd)
	{
		nVector[] layers = new nVector[weights.length];//layer activations of L
		Vector[] deltas = new Vector[weights.length];// deltas of L+1
		boolean deltayet=false;
		int countLayers=0;
		nVector backup=inputs;
		for(WeightLayer w:weights)
		{
			int inlen = inputs.dim;
			layers[countLayers++]=inputs;// store the activations
			if(deltayet)
				deltas[countLayers-2]=new DenseVector(inlen);
			inputs=w.applyWithBias(inputs);
			//inputs=w.apply(inputs);
			backup=inputs;
			if(alternativeSigmoid>0.0)
			{
				inputs=inputs.altSigmoidize();
			} else
				inputs=inputs.sigmoidize();
				
			// outputs of next layer are sent through
			deltayet=true;
		}
		
		Vector in = inputs.data;
		int inlen = in.size();
		
		deltas[deltas.length-1]=new DenseVector(inlen);
		if(epsilon==0.0)
		{
			if(sigmoid)
				return inputs;
			else
				return backup;
		}
		
		
		batchCount++;
		//now we calculate deltas for the entire range

		//delta for last layer
		
		
		Vector delta = sigmoid?expected.data.copy().scale(-1).add(in):expected.data.copy().scale(-1).add(backup.data);
		lossTotal+=delta.norm(Norm.Two);
		if(alternativeSigmoid>0)
		{
			deltas[deltas.length-1] = inputs.derivativeAltSigmoidize().componentWiseProduct(delta);
		} else {
			deltas[deltas.length-1] = inputs.derivativeSigmoidize().componentWiseProduct(delta);
		}
		//a bit more complicated, calculate the deltas for all the other layers (weighted sum)
		for(int layer = weights.length-1;layer>0;layer--)
		{
			WeightLayer weight = weights[layer];
			nVector currentLayer = layers[layer];
			
			nVector dos;
			if(alternativeSigmoid>0)
			{
				dos = currentLayer.derivativeAltSigmoidize();
			} else {
				dos = currentLayer.derivativeSigmoidize();
			}
			
			DenseMatrix resultOfStuff = (DenseMatrix) weight.x.transAmult(new DenseMatrix(deltas[layer]), new DenseMatrix(dos.dim,1));
			Vector error =dos.componentWiseProduct(new DenseVector(resultOfStuff.getData()));
			deltas[layer-1]=error;
		}
		if(batchSize < 1)
		{
			for(int layer=layersBegin;layer<layersEnd;layer++)
			{
				weights[layer].doGradDescentWithBiasImmediate(layers[layer],deltas[layer],epsilon, alternativeSigmoid);
			}
		} else {
			for(int layer=layersBegin;layer<layersEnd;layer++)
			{
				weights[layer].doGradDescentWithBias(layers[layer],deltas[layer],epsilon, alternativeSigmoid);
				//weights[layer].doGradDescent(backupLayers[layer],backupDeltas[layer],epsilon, alternativeSigmoid);
			}
			if(batchCount%batchSize == 0)
			{
				for(int layer=layersBegin;layer<layersEnd;layer++)
					weights[layer].finishGradDescent(batchSize);
			}
		}
		if(sigmoid)
			return inputs;
		else
			return backup;
	}
}