package neuralNet;
/**
 * A wrapper class used to split a neural net into 2 halves, an encoder and decoder
 * @author Zachary
 *
 */
public class Encoder {
	private final NeuralNet encoder;
	private final NeuralNet decoder;
	private final NeuralNet original;
	private final int before;
	//private final int after;
	protected double recentScore=.5f;
	protected double lastScore = 0f;
	protected nVector TrainGenerator(nVector rng, double epsilon, boolean sigmoid, double altSigmoid)
	{
		if(original==null)
		{
			throw new UnsupportedOperationException("cannot train autoencoder like GAN");
		}
		//generate image, then discriminate. Train so it gets near 0
		nVector res = original.trainSpecific(rng, new nVector(new double[]{0}), epsilon, sigmoid, altSigmoid, 0, before);
		double score = res.size();
		recentScore=recentScore*.997f+score*.003f;
		lastScore = score;
		return res;
	}
	protected nVector TrainDiscriminator(nVector input, boolean real, double epsilon, boolean sigmoid, double altSigmoid)
	{
		if(original==null)
		{
			throw new UnsupportedOperationException("cannot train autoencoder like GAN");
		}
		//then discriminate. Calc chances of it being fake
		nVector res = decoder.train(input, new nVector(new double[]{real?0:1}), epsilon, sigmoid, altSigmoid);
		double score = res.subtract(new nVector(new double[]{real?0:1})).size();
		recentScore=recentScore*.997f+score*.003f;
		lastScore = score;
		return res;
	}
	protected double getState()
	{
		return recentScore;
	}
	protected NeuralNet getGenerator()
	{
		if(original==null)
		{
			throw new UnsupportedOperationException("cannot use autoencoder like GAN");
		}
		return encoder;
	}
	protected NeuralNet getDiscriminator()
	{
		if(original==null)
		{
			throw new UnsupportedOperationException("cannot use autoencoder like GAN");
		}
		return decoder;
	}
	protected Encoder(NeuralNet encoder, NeuralNet decoder)
	{
		this.encoder=encoder;
		this.decoder=decoder;
		original=null;
		before=0;
	}
	protected Encoder(NeuralNet original, NeuralNet encoder, NeuralNet decoder, int stepsBefore, int stepsAfter)
	{
		this.encoder=encoder;
		this.decoder=decoder;
		this.original=original;
		before=stepsBefore;
	}
	/**
	 * Calculates the encoded form of the input vector
	 * @param in the vector to encode
	 * @param altsigmoid whether to use leaky RELU instead of sigmoid
	 * @return the encoded vector
	 */
	public nVector encode(nVector in, boolean altsigmoid)
	{
		return encoder.calculate(in, true, altsigmoid);
	}
	/**
	 * Decodes the encoded form of the input vector, generating an output
	 * @param in the vector to decode
	 * @param sigmoid whether or not to sigmoid the output vector
	 * @param altsigmoid whether to use leaky RELU instead of sigmoid
	 * @return the decoded vector
	 */
	public nVector decode(nVector coded, boolean sigmoid, boolean altsigmoid)
	{
		return decoder.calculate(coded, sigmoid, altsigmoid);
	}
}
