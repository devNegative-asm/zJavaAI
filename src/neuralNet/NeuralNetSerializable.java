package neuralNet;

import java.io.Serializable;

final class NeuralNetSerializable implements Serializable {

	/**
	 * 
	 */
	private final int ins;
	private final int outs;
	private final int[] layers;
	private final WeightLayerSerializable[] weights;
	private static final long serialVersionUID = 223L;
	public NeuralNetSerializable(NeuralNet n)
	{
		ins = n.getIns();
		outs = n.getouts();
		layers = n.getInternalLayers();
		weights = n.getWeightSerializable();
	}
	public NeuralNet toNeuralNet()
	{
		return new NeuralNet(ins,outs,layers,weights);
	}
}
