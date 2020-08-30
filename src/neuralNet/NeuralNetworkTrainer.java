package neuralNet;
import java.util.ArrayList;
import java.util.Random;
import java.util.function.Consumer;
/**
 * A class for training a neural network with an exponentially decaying learning rate
 * @author Zachary
 *
 */
public class NeuralNetworkTrainer {
	private NeuralNet coder = null;
	private boolean randomRuns = false;
	private int runs = 1;
	private int repeatsBeforeDownstep = 1;
	private double currentLearningRate = .01;
	private double dipRate = .85;
	private Consumer<NeuralNet> callback = null;
	private double useLinearActivation = 0;
	private int n = 0;
	private Random rng;
	static boolean encoder = false;
	/**
	 *  Constructs a helper class for precicely training a neural network with an exponentially decaying learning rate
	 * 
	 * @param coder The neural network to train
	 * @param randomRuns Whether to pick randomly from training data rather than running all training data in order
	 * @param runs How many times to iterate on the data set before running the callback
	 * @param repeatsBeforeDownstep How many times to call the callback before ajusting the epsilon value
	 * @param originalLearningRate The original value epsilon_0
	 * @param dipRate A double 0<dipRate<=1 to describe what percentage of the original epsilon value to preserve on the downstep. For non-exponential decay, set this to 1 and control decay manually.
	 * @param callback A function to call after running runs iterations
	 * @param useLinearActivation True means use leaky linear activation function, false means use sigmoid function
	 * @param rng A random number generator to use when randomRuns is active 
	 */
	public NeuralNetworkTrainer(NeuralNet coder, boolean randomRuns, int runs, int repeatsBeforeDownstep, double originalLearningRate, double dipRate, Consumer<NeuralNet> callback, double useLinearActivation, Random rng) {
		super();
		this.coder = coder;
		this.randomRuns = randomRuns;
		this.runs = runs;
		this.repeatsBeforeDownstep = repeatsBeforeDownstep;
		this.currentLearningRate = originalLearningRate;
		this.dipRate = dipRate;
		this.callback = callback;
		this.useLinearActivation = useLinearActivation;
		this.rng = rng;
	}
	/**
	 * Checks the loss over the entire data set
	 * @param outputs the list of [dimensionality] number of output vectors
	 * @param dimensionality the dimensionality of the input vectors
	 * @param sigmoidResults whether or not to sigmoid the results
	 * @return a double representing the sum of losses
	 */
	public double getLossOneHot(nVector[] outputs, int dimensionality, boolean sigmoidResults)
	{
		double score = 0;
		for(int y=0;y<dimensionality;y++)
		{
			score += coder.calculate(new nVector(dimensionality,y), sigmoidResults, useLinearActivation>0).subtract(outputs[y]).size();
		}
		return score;
	}
	/**
	 * Checks the loss over the entire data set
	 * @param the input vectors
	 * @param the expected output vectors
	 * @param dimensionality the dimensionality of the input vectors
	 * @param sigmoidResults whether or not to sigmoid the results
	 * @return a double representing the sum of losses
	 */
	public double getLoss(nVector[] inputs, nVector[] outputs, boolean sigmoidResults)
	{
		double score = 0;
		for(int y=0;y<inputs.length;y++)
		{
			score += coder.calculate(inputs[y], sigmoidResults, useLinearActivation>0).subtract(outputs[y]).size();
		}
		return score;
	}
	ArrayList<nVector> fakes = new ArrayList<>();
	public double runFullLoopTrainingGAN(nVector[] inputs, int dimensionalityInput, boolean sigmoidResults, double sigma, int layer)
	{
		coder.setTrainingBatchSize(inputs.length);
		Encoder GAN = coder.getGAN(layer);
		if(fakes.isEmpty())
		{
			for(int i=0;i<1000;i++)
			{
				fakes.add(GAN.encode(nVector.getRandomVector(rng, dimensionalityInput, sigma, 0), useLinearActivation>0));
			}
		} else {
			for(int i=0;i<100;i++)
			{
				fakes.remove(0);
				fakes.add(GAN.encode(nVector.getRandomVector(rng, dimensionalityInput, sigma, 0), useLinearActivation>0));
			}
		}
		GAN.recentScore = 1;
		int cc = 0;
		for(int ind = 0; ind<inputs.length; ind++)
		{
			GAN.TrainDiscriminator(fakes.get(rng.nextInt(fakes.size())), false, currentLearningRate*Math.sqrt(inputs.length), sigmoidResults, useLinearActivation);
		}
		for(int ind = 0; ind<inputs.length; ind++)
		{
			GAN.TrainDiscriminator(inputs[ind], true, currentLearningRate*Math.sqrt(inputs.length), sigmoidResults, useLinearActivation);
		}
		System.out.println("discriminator score:"+GAN.recentScore);
		GAN.recentScore = 1;
		for(int nn=0; GAN.recentScore>.25;nn++)
		{
			for(int ind = 0; ind<inputs.length; ind++)
			{
				GAN.TrainGenerator(nVector.getRandomVector(rng, dimensionalityInput, sigma, 0), currentLearningRate*Math.sqrt(inputs.length), sigmoidResults, useLinearActivation);
			}
			System.out.println("generator score:"+GAN.recentScore);
		}
		return GAN.lastScore;
	}
	/**
	 * 
	 * Trains with [runs] repeats of the entire data set using one-hot inputs, then decreases epsilon if necessary
	 * 
	 * @param outputs The outputs to copy
	 * @param dimensionality Number of dimensions of the one-hot vector
	 * @param sigmoidResults Whether to sigmoid the output layer
	 * @param printEpsilonChanges Whether to print when the epsilon is decayed
	 * @return A double roughtly representing the loss function over the last few results, weighted for more recent results
	 */
	public double runFullLoopTrainingOneHot(nVector[] outputs, int dimensionality, boolean sigmoidResults, boolean printEpsilonChanges)
	{
		coder.trackLoss();
		int ccc = dimensionality;
		for(int nn=0;nn<runs;nn++)
			for(int ind = 0; ind<ccc; ind++)
			{
				int index = randomRuns?rng.nextInt(ccc):ind;
				nVector oneHot = new nVector(ccc,index);
				coder.train(oneHot, outputs[index], currentLearningRate, sigmoidResults, useLinearActivation);
			}
		callback.accept(coder);
		if(++n%repeatsBeforeDownstep==0)
		{
			currentLearningRate*=dipRate;
			if(printEpsilonChanges)
				System.out.println("ep = "+currentLearningRate);
		}
		return coder.getLossAndReset();
	}
	/**
	 * 
	 * Trains with [runs] repeats of the entire data set using the given inputs and outputs, then decreases epsilon if necessary
	 * 
	 * @param outputs The outputs to copy
	 * @param sigmoidResults Whether to sigmoid the output layer
	 * @param printEpsilonChanges Whether to print when the epsilon is decayed
	 * @return A double roughtly representing the loss function over the last few results, weighted for more recent results
	 */
	public double runFullLoopTraining(nVector[] inputs, nVector[] outputs, boolean sigmoidResults, boolean printEpsilonChanges)
	{
		coder.trackLoss();
		int ccc = inputs.length;
		for(int nn=0;nn<runs;nn++)
			for(int ind = 0; ind<ccc; ind++)
			{
				int index = randomRuns?rng.nextInt(ccc):ind;
				coder.train(inputs[index], outputs[index], currentLearningRate, sigmoidResults, useLinearActivation);
			}
		callback.accept(coder);
		if(++n%repeatsBeforeDownstep==0)
		{
			currentLearningRate*=dipRate;
			if(printEpsilonChanges)
				System.out.println("ep = "+currentLearningRate);
		}
		return coder.getLossAndReset();
	}
	boolean isRandomRuns() {
		return randomRuns;
	}
	int getRuns() {
		return runs;
	}
	int getRepeatsBeforeDownstep() {
		return repeatsBeforeDownstep;
	}
	double getCurrentLearningRate() {
		return currentLearningRate;
	}
	double getDipRate() {
		return dipRate;
	}
	Consumer<NeuralNet> getCallback() {
		return callback;
	}
	public void setRandomRuns(boolean randomRuns) {
		this.randomRuns = randomRuns;
	}
	public void setRuns(int runs) {
		this.runs = runs;
	}
	public void setRepeatsBeforeDownstep(int repeatsBeforeDownstep) {
		this.repeatsBeforeDownstep = repeatsBeforeDownstep;
	}
	public void setCurrentLearningRate(double currentLearningRate) {
		this.currentLearningRate = currentLearningRate;
	}
	public void setDipRate(double dipRate) {
		this.dipRate = dipRate;
	}
	public void setCallback(Consumer<NeuralNet> callback) {
		this.callback = callback;
	}
	public void setRng(Random rng) {
		this.rng = rng;
	}
}
