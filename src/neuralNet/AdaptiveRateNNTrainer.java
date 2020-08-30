package neuralNet;
import java.util.Random;
/**
 * A class for training a neural net at a rate determined by which decreases the loss fastest
 * NOTE that unlike NeuralNetworkTrainer, not all of the changes are reflected in the original NeuralNet object.
 * After each training batch, your NeuralNet variable must be reassigned to AdaptiveRateNNTrainer.this.getNeuralNet()
 * @author Zachary
 *
 */
public class AdaptiveRateNNTrainer {
	private NeuralNet coder;
	private boolean randomRuns;
	private int runs;
	private final int repeatsBeforeStep = Integer.MAX_VALUE;
	private double originalLearningRate;
	private double stepRate;
	private double useLinearActivation;
	Random rng;
	/**
	 * Constructor for an AdaptiveRateNNTrainer
	 * @param coder The NeuralNet to train
	 * @param randomRuns whether to train on the input data selected randomly or linearly
	 * @param runs number of times to iterate over the data set before ajusting the learning rate
	 * @param originalLearningRate the original learning rate, ideally set from .05~.1
	 * @param stepRate the rate to change the learning rate. NOTE: on increase, the learning rate is multiplied by 1+stepRate and on decrease by 1-stepRate, so stepRate must be in [0,1) with values closer to 1 decaying much faster
	 * @param useLinearActivation whether to use RELU over the standard sigmoid
	 * @param rng the random number generator to use
	 */
	public AdaptiveRateNNTrainer(NeuralNet coder, boolean randomRuns, int runs, double originalLearningRate, double stepRate, double useLinearActivation, Random rng) {
		this.coder = coder;
		this.randomRuns = randomRuns;
		this.runs = runs;
		this.originalLearningRate = originalLearningRate;
		this.stepRate = stepRate;
		this.useLinearActivation = useLinearActivation;
		this.rng = rng;
	}
	/**
	 * The trained neural net
	 * @return The trained neural net
	 */
	public NeuralNet getNeuralNet()
	{
		return coder;
	}
	/**
	 * Train this neural network as a 1-hot encoder. Run [runs] epochs, then ajust the training rate
	 * Note that some time is spent not learning but predicting the optimal training rate. The ratio of total time to real learning time is
	 * runs : (steps * 3) 
	 * @param outputs the output vectors to train on
	 * @param dimensionality the dimensionality of the input space. Usually, the number of training examples
	 * @param sigmoidResults whether or not to sigmoid the output vectors
	 * @param printEpsilonChanges set to true to log changes in the learning rate to the standard output
	 * @param steps Number of epochs to run in order to predict the optimal learning rate.
	 * @return the loss associated with the best case scenario for [steps] epochs after epsilon ajustment 
	 */
	public double runFullLoopTrainingOneHot(nVector[] outputs, int dimensionality, boolean sigmoidResults, boolean printEpsilonChanges, int steps)
	{
		NeuralNet[] attempts = new NeuralNet[3];
		double[] scores = new double[3];
		double[] rates = new double[3];
		rates[0] = originalLearningRate;
		rates[1] = originalLearningRate*(1+stepRate);
		rates[2] = originalLearningRate*(1-stepRate);
		
		for(int i = 0; i<3;i++)
		{
			final int thread = i;
			final double real_rate = rates[thread];
			NeuralNet net = coder.randomAlter(0);
			attempts[thread] = net;
			NeuralNetworkTrainer tr = new NeuralNetworkTrainer(net, randomRuns, steps, repeatsBeforeStep, real_rate, 1, x->{}, useLinearActivation, rng);
			tr.runFullLoopTrainingOneHot(outputs, dimensionality, sigmoidResults, false);
			scores[thread] = 0;
			for(int y=0;y<outputs.length;y++)
			{
				scores[thread] += net.calculate(new nVector(outputs.length,y), sigmoidResults, useLinearActivation>0).subtract(outputs[y]).size();
			}
			
		}
		double lowestScore = Double.POSITIVE_INFINITY;
		for(int i = 0; i<3;i++)
		{
			if(scores[i]<lowestScore)
			{
				originalLearningRate = rates[i];
				lowestScore = scores[i];
			}
		}
		if(printEpsilonChanges)
		{
			System.out.println("ep = "+originalLearningRate);
		}
		
		NeuralNetworkTrainer coach = new NeuralNetworkTrainer(coder, randomRuns, runs, repeatsBeforeStep, originalLearningRate, 1, x->{}, useLinearActivation, rng);
		coach.runFullLoopTrainingOneHot(outputs, dimensionality, sigmoidResults, false);
		
		
		return lowestScore;
	}
	/**
	 * Train this as a standard neural network. Run [runs] epochs, then ajust the training rate
	 * Note that some time is spent not learning but predicting the optimal training rate. The ratio of total time to real learning time is
	 * (runs + steps) : (steps * 3) 
	 * @param inputs the input vectors
	 * @param outputs the output vectors to train on
	 * @param sigmoidResults whether or not to sigmoid the output vectors
	 * @param printEpsilonChanges set to true to log changes in the learning rate to the standard output
	 * @param steps Number of epochs to run in order to predict the optimal learning rate.
	 */
	public double runFullLoopTraining(nVector[] inputs, nVector[] outputs, boolean sigmoidResults, boolean printEpsilonChanges, int steps)
	{
		NeuralNet[] attempts = new NeuralNet[3];
		double[] scores = new double[3];
		double[] rates = new double[3];
		rates[0] = originalLearningRate;
		rates[1] = originalLearningRate*(1+stepRate);
		rates[2] = originalLearningRate*(1-stepRate);
		
		NeuralNetworkTrainer coach = new NeuralNetworkTrainer(coder, randomRuns, runs, repeatsBeforeStep, originalLearningRate, 1, x->{}, useLinearActivation, rng);
		coach.runFullLoopTraining(inputs, outputs, sigmoidResults, false);
		
		for(int i = 0; i<3;i++)
		{
			final int thread = i;
			final double real_rate = rates[thread];
			NeuralNet net = coder.randomAlter(0);
			attempts[thread] = net;
			NeuralNetworkTrainer tr = new NeuralNetworkTrainer(net, randomRuns, steps, repeatsBeforeStep, real_rate, 1, x->{}, useLinearActivation, rng);
			tr.runFullLoopTraining(inputs, outputs, sigmoidResults, false);
			for(int y=0;y<outputs.length;y++)
			{
				scores[thread] += net.calculate(inputs[y], sigmoidResults, useLinearActivation>0).subtract(outputs[y]).size();
			}
		}
		double lowestScore = Double.POSITIVE_INFINITY;
		for(int i = 0; i<3;i++)
		{
			if(scores[i]<lowestScore)
			{
				originalLearningRate = rates[i];
				lowestScore = scores[i];
			}
		}
		if(printEpsilonChanges)
		{
			System.out.println("ep = "+originalLearningRate);
		}
		return lowestScore;
	}
	/**
	 * Gets the current learning rate
	 * @return the learning rate
	 */
	public double getLearningRate()
	{
		return originalLearningRate;
	}
}
