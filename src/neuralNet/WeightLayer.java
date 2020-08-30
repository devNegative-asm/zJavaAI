package neuralNet;
import java.io.PrintWriter;
import java.util.Random;

import Jama.Matrix;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.Vector.Norm;
/**
 * A matrix used either for PCA or as a neural connection between layers of a Neural Network
 * @author Zachary
 */
public class WeightLayer {
	
	DenseMatrix x;
	int ins;
	int outs;
	DenseVector bias;
	DenseMatrix dx;
	DenseVector db;
	/**
	 * @deprecated
	 * Returns a vector representation of this weight layer
	 * @return
	 */
	public nVector toVector()
	{
		double[] matrixValues = x.getData();
		double[] biasValues = bias.getData();
		double[] vals = new double[matrixValues.length+biasValues.length];
		System.arraycopy(matrixValues, 0, vals, 0, matrixValues.length);
		System.arraycopy(biasValues, 0, vals, matrixValues.length, biasValues.length);
		return new nVector(vals);
	}
	
	/**
	 * Whether or not this WeightLayer has NANs or Infinities
	 * @return true if no NANs or Infinities are found, else false
	 */
	public boolean safe()
	{
		for(double d:x.getData())
		{
			if(!Double.isFinite(d))
				return false;
		}
		for(double d:bias.getData())
		{
			if(!Double.isFinite(d))
				return false;
		}
		return true;
	}
	/**
	 * Returns a deep copy of this WeightLayer
	 */
	public WeightLayer clone()
	{
		return new WeightLayer(x.copy(), bias.copy());
	}
	/**
	 * Constructs a WeightLayer given an mtj Matrix as its connections and a Vector as its bias
	 * @param x The matrix representing the neurons
	 * @param bias a bias vector added to outputs
	 */
	public WeightLayer(DenseMatrix x, DenseVector bias) {
		this.x=x;
		outs = x.numRows();
		ins = x.numColumns();
		if(bias ==null)
		{
		bias = new DenseVector(outs);
		bias.zero();
		} else {
			this.bias=bias;
		}
		dx = new DenseMatrix(outs,ins);
		db = new DenseVector(outs);
	}
	WeightLayer(int ins, int outs, double sigma, double mean, Random rng)
	{
		double[][] weights = new double[outs][ins];
		bias = new DenseVector(outs);
		for(int i=0;i<outs;i++)
		{
			bias.set(i,(rng.nextDouble()-.5)*sigma+mean);
			for(int j=0;j<ins;j++)
			{
				weights[i][j]=(double) ((rng.nextDouble()-.5)*sigma/Math.sqrt(ins)+mean);
			}
		}
		if(outs!=0&&ins!=0)
			x = new DenseMatrix(weights);
		this.ins=ins;
		this.outs=outs;
		dx = new DenseMatrix(outs,ins);
		db = new DenseVector(outs);
	}
	protected void WriteToStream(PrintWriter printer)
	{
		printer.print(ins);
		printer.print("A");
		printer.print(outs);
		printer.print("A");
		for(int row = 0; row < outs -1; row++)
		{
			for(int col = 0; col < ins-1; col++)
			{
				printer.print(x.get(row, col));
				printer.print('C');
			}
			printer.print(x.get(row, ins-1));
			printer.print('B');
		}
		for(int col = 0; col < ins-1; col++)
		{
			printer.print(x.get(outs-1, col));
			printer.print('C');
		}
		printer.print(x.get(outs-1, ins-1));
		printer.print("A");
		for(int row = 0; row < outs-1; row++)
		{
			printer.print(bias.get(row));
			printer.print('C');
		}
		printer.print(bias.get(outs-1));
	}
	/**
	 * Returns a full representation of this WeightLayer as a string. Only use for saving a neural net as this can easily spam the terminal if printed. 
	 */
	public String toString()
	{
		StringBuilder retval = new StringBuilder();
		//divide major segments with A
		//divide lists with B
		//divide items of a list with C
		retval.append(ins);
		retval.append('A');
		retval.append(outs);
		retval.append('A');
		//now we have some work to do.
		for(int row = 0; row < outs; row++)
		{
			for(int col = 0; col < ins; col++)
			{
				retval.append(x.get(row, col));
				retval.append('C');
			}
			retval.deleteCharAt(retval.length()-1);//dont need a C in the way.
			retval.append('B');
		}
		retval.deleteCharAt(retval.length()-1);//dont need a B in the way.
		retval.append('A');
		for(int row = 0; row < outs; row++)
		{
			retval.append(bias.get(row));
			retval.append('C');
		}
		retval.deleteCharAt(retval.length()-1);//dont need a B in the way.
		return retval.toString();
	}
	/**
	 * Reverses to toString method and constructs a WeightLayer from its string representation
	 * @param s the formatted string
	 * @return a WeightLayer with the loaded data
	 */
	static WeightLayer fromString(String s)
	{
		String[] ad=s.split("A");
		String[] elements2=ad[3].split("C");
		DenseMatrix x = new DenseMatrix(Integer.parseInt(ad[1]),Integer.parseInt(ad[0]));
		s=ad[2];
		ad=s.split("B");
		int outerTarget=0;
		for(String bb:ad)
		{
			int innerTarget=0;
			//these are each of the lists.
			String[] elements = bb.split("C");
			for(String elem:elements)
			{
				x.set(outerTarget, innerTarget++, Double.parseDouble(elem));
			}
			outerTarget++;
		}
		DenseVector bias = new DenseVector(elements2.length);
		outerTarget=0;
		for(String elem:elements2)
		{
			bias.set(outerTarget++, Double.parseDouble(elem));
		}
		return new WeightLayer(x, bias);
	}
	
	
	/**
	 * Performs gradient descent with the given activation and delta vectors, clipping at threshold
	 * @param activations The activations of the previous layer
	 * @param deltas The deltas of the subsequent layer
	 * @param epsilon The training rate
	 * @param threshold the maximum norm acceptable for a gradient
	 */
	void doGradDescent(nVector activations,Vector deltas, double epsilon, double threshold)
	{	
		double norm = deltas.norm(Norm.Two);
		// This matrix +=  -epsilon times the outer product of activations & deltas 
		if(threshold>0&&norm>threshold)
		{
			deltas.scale(threshold/norm);
		}
		//x.rank1(-epsilon, activations.data, deltas.data);
		//^ that would work only if x is square, but it usually isn't
		new DenseMatrix(deltas).multAdd(-epsilon, new DenseMatrix(activations.data).transpose(new DenseMatrix(1,activations.data.size())), dx);
		// weights[to][from]
		// info has from encoded in value
		// info has to encoded in score
		
	}
	/**
	 * Performs gradient descent with the given activation and delta vectors and updates the biases, clipping at threshold
	 * @param activations The activations of the previous layer
	 * @param deltas The deltas of the subsequent layer
	 * @param epsilon The training rate
	 * @param threshold the maximum norm acceptable for a gradient
	 */
	void doGradDescentWithBias(nVector activations,Vector deltas, double epsilon, double threshold)
	{
		// weights[to][from]
		// info has from encoded in value
		// info has to encoded in score
		double norm = deltas.norm(Norm.Two);
		// This matrix +=  -epsilon times the outer product of activations & deltas 
		if(threshold>0&&norm>threshold)
		{
			deltas.scale(threshold/norm);
		}

		new DenseMatrix(deltas).multAdd(-epsilon, new DenseMatrix(activations.data).transpose(new DenseMatrix(1,activations.data.size())), dx);
		db.add(-epsilon, deltas);
	}
	/**
	 * Immediately performs gradient descent with the given activation and delta vectors, clipping at threshold
	 * @param activations The activations of the previous layer
	 * @param deltas The deltas of the subsequent layer
	 * @param epsilon The training rate
	 * @param threshold the maximum norm acceptable for a gradient
	 */
	void doGradDescentImmediate(nVector activations,Vector deltas, double epsilon, double threshold)
	{	
		double norm = deltas.norm(Norm.Two);
		// This matrix +=  -epsilon times the outer product of activations & deltas 
		if(threshold>0&&norm>threshold)
		{
			deltas.scale(threshold/norm);
		}
		//x.rank1(-epsilon, activations.data, deltas.data);
		//^ that would work only if x is square, but it usually isn't
		new DenseMatrix(deltas).multAdd(-epsilon, new DenseMatrix(activations.data).transpose(new DenseMatrix(1,activations.data.size())), x);
		// weights[to][from]
		// info has from encoded in value
		// info has to encoded in score
		
	}
	/**
	 * Immediately performs gradient descent with the given activation and delta vectors and updates the biases, clipping at threshold
	 * @param activations The activations of the previous layer
	 * @param deltas The deltas of the subsequent layer
	 * @param epsilon The training rate
	 * @param threshold the maximum norm acceptable for a gradient
	 */
	void doGradDescentWithBiasImmediate(nVector activations,Vector deltas, double epsilon, double threshold)
	{
		// weights[to][from]
		// info has from encoded in value
		// info has to encoded in score
		double norm = deltas.norm(Norm.Two);
		// This matrix +=  -epsilon times the outer product of activations & deltas 
		if(threshold>0&&norm>threshold)
		{
			deltas.scale(threshold/norm);
		}
		new DenseMatrix(deltas).multAdd(-epsilon, new DenseMatrix(activations.data).transpose(new DenseMatrix(1,activations.data.size())), x);
		bias.add(-epsilon, deltas);
	}
	/**
	 * Multiplies the given vector by this weightlayer in a manner which "feeds it through" the network
	 * @param v the activations vector
	 * @return the vector which is this matrix * v
	 */
	public nVector apply(nVector v)
	{
		Vector out = new DenseVector(outs);
		return new nVector(x.mult(v.data, out));
	}
	/**
	 * Multiplies the given vector by this weightlayer in a manner which "feeds it through" the network, then adds the bias
	 * @param v the activations vector
	 * @return the vector which is this (matrix * v) + bias
	 */
	public nVector applyWithBias(nVector v)
	{
		Vector out = bias.copy();
		return new nVector(x.multAdd(1, v.data, out));
	}
	protected void finishGradDescent(int count)
	{
		x.add(dx);
		bias.add(db);
		dx.zero();
		db.zero();
	}
	public nVector inverseMult(nVector in)
	{
		in = in.subtract(new nVector(bias));
		Matrix myinv = new Matrix(x.getData(),x.numRows()).inverse();
		Matrix results = myinv.times(new Matrix(new DenseVector(in.getDataCopy()).getData(),ins));
		return new nVector(results.getColumnPackedCopy());
	}
}