package neuralNet;

import java.io.Serializable;

import Jama.Matrix;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;

final class WeightLayerSerializable implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 333L;
	private final Matrix x;
	private final DenseVector bias;
	protected WeightLayerSerializable(WeightLayer wl)
	{
		this.x = new Matrix(wl.x.getData(),wl.x.numRows());
		bias = wl.bias;
	}
	protected WeightLayer toWeightLayer()
	{
		return new WeightLayer(new DenseMatrix(x.getArray()),bias);
	}
}
