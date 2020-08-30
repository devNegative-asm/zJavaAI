package neuralNet;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import java.util.function.Function;


/**
 * A class used to interact with the neural net related classes in a more readable format
 * @author Zachary
 *
 */
public class NeuralNetHelper<I,O>{
	/**
	 * Sets up a standard nVector-in nVector out neural network.
	 * This is useful for when you have batches of objects which are easier to convert in bulk
	 */
	public static <X extends Vectorizable<X>,Y extends Vectorizable<Y>> NeuralNetHelper<X,Y>.NeuralNetHelper1.NeuralNetHelper2 defaultConfig(Function<nVector,Y> reverser)
	{
		return new NeuralNetHelper<X,Y>().setInputFunction(X::toVector).setOutputFunction(reverser, Y::toVector);
	}
	private NeuralNetHelper<I,O>.NeuralNetHelper1.NeuralNetHelper2.NeuralNetHelper3.NeuralNetHelper4.NeuralNetHelper5.NeuralNetHelper6.NeuralNetHelper7.NeuralNetHelper8.NeuralNetHelper9.NeuralNetHelper10.NeuralNetHelper11.NeuralNetHelper12.NeuralNetHelper13 x;
	public NeuralNetHelper()
	{}
	private NeuralNetHelper(NeuralNetHelper<I,O>.NeuralNetHelper1.NeuralNetHelper2.NeuralNetHelper3.NeuralNetHelper4.NeuralNetHelper5.NeuralNetHelper6.NeuralNetHelper7.NeuralNetHelper8.NeuralNetHelper9.NeuralNetHelper10.NeuralNetHelper11.NeuralNetHelper12.NeuralNetHelper13 x)
	{
		this.x=x;
	}
	/**
	 * Get the neural net in the container.
	 * @return the neural net
	 */
	public NeuralNet getNeuralNet()
	{
		return x.getCurrentNeuralNet();
	}
	/**
	 * Runs the neural net on the provided input and generates an output
	 * @param in The input object
	 * @param useSigmoid whether the neural net should sigmoid its outputs
	 * @return the output of the neural network, turned back into an object of type O.
	 */
	public O calculate(I in, boolean useSigmoid)
	{
		return x.calculate(in, useSigmoid);
	}
	/**
	 * Trains the neural net on the provided input and generates an output
	 * @param in The input object
	 * @param useSigmoid whether the neural net should sigmoid its outputs
	 * @return the output of the neural network, turned back into an object of type O.
	 */
	public O trainOneExample(I in, O out, boolean useSigmoid)
	{
		return x.trainOneExample(in, out, useSigmoid);
	}
	/**
	 * Trains the neural network on an array of examples
	 * @param ins an array of inputs
	 * @param outs and array of outputs the net should try to fit
	 * @param useSigmoid whether the neural net should sigmoid its outputs
	 * @param printEpsilonChanges whether or not changes in the learning rate should be logged to System.out
	 * @param predictionDepth How many epochs to run to determine next learning rate. These epochs do no training, so keep this low
	 * @return the sum of loss
	 */
	public double train(I[] ins, O[] outs, boolean useSigmoid, boolean printEpsilonChanges, int predictionDepth)
	{
		return x.train(ins, outs, useSigmoid, printEpsilonChanges, predictionDepth);
	}
	/**
	 * Supply a function which turns one of the input type into a vector
	 */
	public NeuralNetHelper1 setInputFunction(Function<I,nVector> inputFunction)
	{
		return new NeuralNetHelper1(inputFunction);
	}
	public class NeuralNetHelper1{
		private Function<I,nVector> inputFunction;
		NeuralNetHelper1(Function<I, nVector> inputFunction) {
			this.inputFunction = inputFunction;
		}
		/**
		 * Supply a function which turns a vector into one of the output type, and a function which turns one of the output type into a vector to use with least squares loss
		 */
		public NeuralNetHelper2 setOutputFunction(Function<nVector,O> vecToOutFunction, Function<O,nVector> outToVecFunction)
		{
			return new NeuralNetHelper2(vecToOutFunction,outToVecFunction );
		}
		public class NeuralNetHelper2{
			private Function<nVector,O> outputFunction;
			private Function<O,nVector> outputFunction2;
			private boolean useLoaded = false;
			private NeuralNet fill;
			NeuralNetHelper2(Function<nVector,O> outputFunction1, Function<O,nVector> outputFunction2) {
				this.outputFunction = outputFunction1;
				this.outputFunction2=outputFunction2;
			}
			/** 
			 * Supply an integer which represents the dimensionality of the input vector
			 */
			public NeuralNetHelper3 setInputDimension(int inputDimension)
			{
				return new NeuralNetHelper3(inputDimension);
			}
			/**
			 * Supply a file containing a saved neural network
			 * @param f a file to which a neural network has been written
			 * @return a configuration where the neural net settings are auto detected and auto initialized
			 * @throws IOException when reading the file fails
			 * @throws ClassNotFoundException when the data cannot be read properly
			 */
			public NeuralNetHelper3.NeuralNetHelper4.NeuralNetHelper5 fromFile(File f) throws ClassNotFoundException, IOException
			{
				useLoaded = true;
				fill = NeuralNet.fromFile(f);
				return this
						.setInputDimension(fill.getIns())
						.setOutputDimension(fill.getouts())
						.setHiddenLayerDimensions(fill.getInternalLayers());
			}
			public class NeuralNetHelper3{
				private int inputDim;
				NeuralNetHelper3(int inputDim) {
					this.inputDim = inputDim;
				}
				/**
				 * Supply an integer which represents the dimensionality of the output vector
				 */
				public NeuralNetHelper4 setOutputDimension(int outputDimension)
				{
					return new NeuralNetHelper4(outputDimension);
				}
				public class NeuralNetHelper4{
					private int outDim;
					NeuralNetHelper4(int outputDimension) {
						outDim=outputDimension;
					}
					/**
					 * Supply an array of ints representing the dimensions of each hidden layer.
					 * The depth of the neural network is hiddenDims.length
					 * and the number of hidden nodes is the sum of the elements of hiddenDims
					 */
					public NeuralNetHelper5 setHiddenLayerDimensions(int[] hiddenDims)
					{
						return new NeuralNetHelper5(hiddenDims);
					}
					public class NeuralNetHelper5{
						private int[] hiddenD;
						NeuralNetHelper5(int[] hiddenDims) {
							this.hiddenD=hiddenDims;
						}
						/**
						 * Supply a random number generator for random functions
						 */
						public NeuralNetHelper6 setRandomNumberGenerator(Random rng)
						{
							return new NeuralNetHelper6(rng);
						}
						public class NeuralNetHelper6{
							private Random rng;
							NeuralNetHelper6(Random rng) {
								this.rng = rng;
							}
							/**
							 * Supply the mean and standard deviation for the initial weights of the neural network.
							 * To initialize all weights to 0, supply 0,0
							 */
							public NeuralNetHelper7 setWeightsGaussian(double mean, double sigma)
							{
								return new NeuralNetHelper7(mean, sigma);
							}
							public class NeuralNetHelper7{
								private double mean,sigma;
								NeuralNetHelper7(double mean, double sigma) {
									this.mean = mean;
									this.sigma = sigma;
								}
								/**
								 * Supply true if you wish to train on random samples from the data set
								 * Supply false if you wish to train over the entire dataset in order
								 */
								public NeuralNetHelper8 setTrainRandomly(boolean trainRandomly){
									return new NeuralNetHelper8(trainRandomly);
								}
								public class NeuralNetHelper8{
									private boolean random;
									NeuralNetHelper8(boolean randomly) {
										this.random = randomly;
									}
									/**
									 * Supply the number of epochs the neural net should train for before ajusting its learning rate
									 */
									public NeuralNetHelper9 setEpochsPerAjustment(int epochs)
									{
										return new NeuralNetHelper9(epochs);
									}
									public class NeuralNetHelper9{
										private int epochs ;
										NeuralNetHelper9(int epoch) {
											this.epochs = epoch;
										}
										/**
										 * Supply the starting learning rate
										 * The learning rate should usually start somewhere from .05~.1
										 */
										public NeuralNetHelper10 setStartingLearningRate(double rate)
										{
											return new NeuralNetHelper10(rate);
										}
										public class NeuralNetHelper10{
											private double rate0 ;
											NeuralNetHelper10(double rate) {
												this.rate0 = rate;
											}
											/**
											 * Supply the rate at which the learning rate changes.
											 * For increasing, the learning rate is multiplied by rate+1
											 * For decreasing, the learning rate is multiplied by rate-1
											 * rate must be in the range [0,1) and values close to but above 0 are generally better
											 */
											public NeuralNetHelper11 setLearningRateAjustment(double rate)
											{
												if(rate<0||rate>=1)
													throw new ArithmeticException("rate must be in interval [0,1)");
												return new NeuralNetHelper11(rate);
											}
											public class NeuralNetHelper11{
												private double ajustmentRate;
												NeuralNetHelper11(double rate) {
													this.ajustmentRate = rate;
												}
												/**
												 * Supply a value over 0 to use leaky RELU instead of the standard sigmoid
												 * a non-zero value will set the threshold for applying the gradient
												 */
												public NeuralNetHelper12 setReluUse(double threshold)
												{
													return new NeuralNetHelper12(threshold);
												}
												public class NeuralNetHelper12{
													private double useRelu;
													NeuralNetHelper12(double threshold) {
														this.useRelu = threshold;
													}
													/**
													 * Supply a momentum value for gradient descent
													 * Constructs a neural net container (net + trainer) based on the provided constraints
													 * @return a neural net container matching the constraints
													 */
													public NeuralNetHelper13 setMomentum(double momentum)
													{
														return new NeuralNetHelper13();
													}
													public class NeuralNetHelper13{
														private NeuralNet net;
														private AdaptiveRateNNTrainer trainer;
														NeuralNetHelper13()
														{
															if(useLoaded)
															{
																net = fill;
															} else
																net = new NeuralNet(inputDim,hiddenD,outDim,sigma,mean,rng);
															trainer = new AdaptiveRateNNTrainer(net, random, epochs, rate0, ajustmentRate, useRelu, rng);
															
														}
														public NeuralNetHelper<I,O> make()
														{
															return new NeuralNetHelper<I, O>(this);
														}
														/**
														 * Get the neural net currently in the container.
														 * Note: training via this container will change the neural net reference,
														 * so make sure to update your reference by another call to this method after training
														 * @return
														 */
														NeuralNet getCurrentNeuralNet()
														{
															return net;
														}
														nVector[] vectorizeInputs(I[] ins)
														{
															nVector[] ret = new nVector[ins.length];
															for(int i=0;i<ins.length;i++)
															{
																ret[i] = inputFunction.apply(ins[i]);
															}
															return ret;
														}
														nVector[] vectorizeOutputs(O[] ins)
														{
															nVector[] ret = new nVector[ins.length];
															for(int i=0;i<ins.length;i++)
															{
																ret[i] = outputFunction2.apply(ins[i]);
															}
															return ret;
														}
														/*
														private O[] devectorizeOutputs(nVector[] outs)
														{
															O[] ret = (O[]) new Object[outs.length];
															for(int i=0;i<outs.length;i++)
															{
																ret[i] = outputFunction.apply(outs[i]);
															}
															return ret;
														}
														*/
														/**
														 * Runs the neural net on the provided input and generates an output
														 * @param in The input object
														 * @param useSigmoid whether the neural net should sigmoid its outputs
														 * @return the output of the neural network, turned back into an object of type O.
														 */
														O calculate(I in, boolean useSigmoid)
														{
															return outputFunction.apply(net.calculate(inputFunction.apply(in), useSigmoid, useRelu>0));
														}
														/**
														 * Trains the neural net on the provided input and generates an output
														 * @param in The input object
														 * @param useSigmoid whether the neural net should sigmoid its outputs
														 * @return the output of the neural network, turned back into an object of type O.
														 */
														O trainOneExample(I in, O out, boolean useSigmoid)
														{
															double optimalRate = trainer.getLearningRate();
															nVector ex= net.train(inputFunction.apply(in),outputFunction2.apply(out), optimalRate, useSigmoid, useRelu);
															return outputFunction.apply(ex);
														}
														/**
														 * Trains the neural network on an array of examples
														 * @param ins an array of inputs
														 * @param outs and array of outputs the net should try to fit
														 * @param useSigmoid whether the neural net should sigmoid its outputs
														 * @param printEpsilonChanges whether or not changes in the learning rate should be logged to System.out
														 * @param predictionDepth How many epochs to run to determine next learning rate. These epochs are 1/3 as efficient, so keep this low
														 * @return the sum of loss
														 */
														double train(I[] ins, O[] outs, boolean useSigmoid, boolean printEpsilonChanges, int predictionDepth)
														{
															double score = trainer.runFullLoopTraining(vectorizeInputs(ins), vectorizeOutputs(outs), useSigmoid, printEpsilonChanges, predictionDepth);
															return score;
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
