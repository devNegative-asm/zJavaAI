package neuralNet;
import java.util.Random;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.Vector.Norm;
import no.uib.cipr.matrix.sparse.SparseVector;
/**
 * A standard vector with operations defined for use in a neural net
 * @author Zachary
 *
 */
public class nVector implements Vectorizable<nVector>{
	public final int dim;
	Vector data;
	private double[] scores;
	public final boolean oneHot;
	public final int oneHotIndex;
	/**
	 * Calculates the average of the input vector array
	 * @param values the input vectors
	 * @return the average vector, calculated component-wise
	 */
	public static nVector average(nVector[] values)
	{
		nVector mean = new nVector(values[0].dim);
		for(nVector vv:values)
		{
			mean = mean.add(vv);
		}
		mean = mean.multiply(1.0/values.length);
		return mean;
	}
	/**
	 * Performs singular value decomposition on the input vectors and returns a matrix which maps into the principal components
	 * @param values the values with which to perform SVD. It is required that values.length > values[n].dim 
	 * @return the WeightLayer which represents the matrix of principal components, where bias is set to average(values)
	 */
	public static WeightLayer SVD(nVector[] values)
	{
		Matrix X = new Matrix(values.length, values[0].dim);
		int i=0;
		nVector avg = average(values);
		Vector average = avg.getDataCopy();
		Vector negativeaverage = avg.getDataCopy();
		negativeaverage.scale(-1);
		for(nVector vv:values)
		{
			DenseVector nextRow = new DenseVector(vv.getDataCopy());
			
			nextRow.add(negativeaverage);// negative mean
			Matrix row = new Matrix(nextRow.getData(),1);
			X.setMatrix(i,i,0,values[0].dim-1,row);
			i++;
		}
		//based around 0 now
		//out of USV*, we need SV*
		SingularValueDecomposition svd = X.svd();
		X =(svd.getV().times(svd.getS())); // (SV*)* = (V**S) = VS
		return new WeightLayer(new DenseMatrix(X.getArrayCopy()),new DenseVector(average));
	}
	/**
	 * Gets a (deep) copy of the the mtj representation of this vector
	 * @return a mtj representation of this vector 
	 */
	public Vector getDataCopy()
	{
		return data.copy();
	}
	/**
	 * Sets a component of this vector
	 * @param d the dimension of the component
	 * @param val what to set the component to
	 */
	public void setComponent(int d, double val)
	{
		data.set(d,val);
	}
	/**
	 * Calculates this * v as a component wise product
	 * @param v the vector to multiply
	 * @return this * v
	 */
	public nVector componentWiseProduct(nVector v)
	{
		if(this.dim!=v.dim)
			throw new ArithmeticException("dimension mismatch in cwp");
		double[] dataNew = new double[dim];
		for(int i=0;i<dim;i++)
		{
			dataNew[i]=data.get(i)*v.data.get(i);
		}
		return new nVector(dataNew);
	}
	/**
	 * Calculates this * v as a component wise product
	 * @param v the vector to multiply
	 * @return this * v
	 */
	public Vector componentWiseProduct(Vector v)
	{
		if(this.dim!=v.size())
			throw new ArithmeticException("dimension mismatch in cwp");
		double[] dataNew = new double[dim];
		for(int i=0;i<dim;i++)
		{
			dataNew[i]=data.get(i)*v.get(i);
		}
		return new DenseVector(dataNew);
	}
	/**
	 * Returns a random vector distributed around mean with standard deviation sigma in each dimension
	 * @param rng the random number generator
	 * @param dim the number of dimensions in the result vector
	 * @param sigma the standard deviation of the distribution in each dimension
	 * @param mean the mean of the distribution in each dimension
	 * @return A random vector
	 */
	public static nVector getRandomVector(Random rng, int dim, double sigma, double mean)
	{
		nVector v = new nVector(dim);
		for(int i=0;i<dim;i++)
		{
			v.data.set(i,sigma*rng.nextGaussian()+mean);
		}
		return v;
	}
	/**
	 * returns an orthogonal basis of the hyperplane spanned by the input vectors.
	 * Note that the output vectors will not necessarily be normalized and the second instance of a collinear vector will be set to either the zero vector or a vector close to it.
	 * @param input the vectors representing the hyperplane
	 * @return a set of orthogonal vectors which span the same hyperplane
	 */
	public static nVector[] getOrthogonalBasis(nVector[] input)
	{
		if(input.length==0)
			return new nVector[0];
		
		nVector average=average(input);
		
		nVector[] output = new nVector[input.length];
		
		for(int nVector=0;nVector<input.length;nVector++)
		{
			nVector v = new nVector(input[nVector].data.copy());
			v=v.subtract(average);
			for(int vsub=0;vsub<nVector;vsub++)
			{
				v=v.subtract(output[vsub].multiply(output[vsub].dot(v)));
			}
			output[nVector]=v;
		}
		return output;
	}
	/**
	 * The dot product this * b, also represented as this.transpose*b
	 * @param b the vector with which to calculate the dot product
	 * @return this * b in terms of the dot product
	 */
	public double dot(nVector b)
	{
		if(dim!=b.dim)
			throw new ArithmeticException("cannot dot vectors of size "+dim+" and "+b.dim+".");
		return data.dot(b.data);
	}
	/**
	 * Calculates the vector this*c where c is taken as a scalar
	 * @param c the scalar with which to scale this vector
	 * @return this*c as a scaled vector
	 */
	public nVector multiply(double c)
	{
		return new nVector(data.copy().scale(c));
	}
	/**
	 * The negative sum this + (-b)
	 * @param b the vector with which to calculate the difference
	 * @return this - b in terms of vector subtraction
	 */
	public nVector subtract(nVector b)
	{
		if(dim!=b.dim)
			throw new ArithmeticException("cannot add vectors of size "+dim+" and "+b.dim+".");
		return new nVector(b.data.copy().scale(-1).add(data));
	}
	/**
	 * The sum this + b
	 * @param b the vector with which to calculate the sum
	 * @return this + b in terms of vector addition
	 */
	public nVector add(nVector b)
	{
		if(dim!=b.dim)
			throw new ArithmeticException("cannot add vectors of size "+dim+" and "+b.dim+".");
		return new nVector(b.data.copy().add(data));
	}
	/**
	 * Calculates the sigmoid function of b = 1/(1+e^-b)
	 * @param b the double to sigmoid
	 * @return sigmoid(b)
	 */
	public static double sigmoid(double b)
	{
		return (double) (1f/(Math.exp(-b)+1f));
	}
	/**
	 * Calculates RELU of b
	 * @param b the input for which to calculate the ELU activation
	 * @return RELU of b
	 */
	public static double altSigmoid(double b)
	{
		return Math.log1p(Math.exp(b));
	}
	/**
	 * Calculates the derivative of the RELU function
	 * @param b location to differentiate
	 * @return the derivative
	 */
	public static double derivativeAltSigmoid(double b)
	{
		return 1-Math.exp(-b);
	}
	/**
	 * Calculates the derivative of the sigmoid function
	 * @param b location to differentiate
	 * @return the derivative
	 */
	public static double derivativeSigmoid(double b)
	{
		return b*(1-b);
	}
	/**
	 * Creates a vector where each component is the sigmoid of the component in this vector
	 * @return sigmoid(this) component-wise
	 */
	public nVector sigmoidize()
	{
		Vector bb = data.copy();
		for(int i = 0; i<bb.size();i++)
		{
			bb.set(i, sigmoid(bb.get(i)));
		}
		return new nVector(bb);
	}
	/**
	 * Creates a vector where each component is the RELU of the component in this vector
	 * @return RELU(this) component-wise
	 */
	public nVector altSigmoidize()
	{
		Vector bb = data.copy();
		for(int i = 0; i<bb.size();i++)
		{
			bb.set(i, altSigmoid(bb.get(i)));
		}
		return new nVector(bb);
	}
	nVector derivativeSigmoidize()
	{
		Vector bb = data.copy();
		for(int i = 0; i<bb.size();i++)
		{
			bb.set(i, derivativeSigmoid(bb.get(i)));
		}
		return new nVector(bb);
	}
	nVector derivativeAltSigmoidize()
	{
		Vector bb = data.copy();
		for(int i = 0; i<bb.size();i++)
		{
			bb.set(i, derivativeAltSigmoid(bb.get(i)));
		}
		return new nVector(bb);
	}
	/**
	 * The euclidean distance from this vector to the origin
	 * @return the distance
	 */
	public double size()
	{
		double size =data.norm(Norm.TwoRobust); //Two means euclidean distance
		return size;
	}
	/**
	 * Creates a 1-hot vector for use in feature extraction.
	 * This is more efficient due to the ability to save sparse data
	 * @param dim the dimensionality of the input space
	 * @param oneHot which dimension to activate
	 */
	public nVector(int dim, int oneHot)
	{
		this.oneHot = true;
		this.dim = dim;
		data=new SparseVector(dim);
		scores= new double[dim];
		data.set(oneHot, 1d);
		oneHotIndex = oneHot;
	}
	/**
	 * Creates a standard vector
	 * @param dim the dimensionality of the vector
	 */
	public nVector(int dim){
		this.dim=dim;
		this.oneHot = false;
		oneHotIndex = 0;
		data=new DenseVector(dim);
		scores= new double[dim];
	}
	/**
	 * Creates a standard vector as a copy of components
	 * @param components the values to copy into this vector
	 */
	public nVector(double[] components){
		this.oneHot = false;
		oneHotIndex = 0;
		this.dim=components.length;
		data= new DenseVector(components.clone());
		scores= new double[dim];
	}
	/**
	 * Creates a standard vector as a copy of scale (an mtj vector)
	 * @param scale the vector to copy into this vector
	 */
	nVector(Vector scale) {
		this.data=scale;
		dim = scale.size();
		oneHotIndex = 0;
		oneHot = false;
		scores= new double[dim];
	}
	/**
	 * copy constructor
	 * @param in vector to copy
	 */
	public nVector(nVector in)
	{
		this.data=in.getDataCopy();
		dim = in.data.size();
		oneHotIndex = 0;
		oneHot = false;
		scores= new double[dim];
	}
	void setScore(int index,double value)
	{
		scores[index]=value;
	}
	/**
	 * Set the value of this[index] = value
	 */
	public void setValue(int index, double value)
	{
		data.set(index, value);
	}
	double getScore(int index)
	{
		return scores[index];
	}
	/**
	 * Returns the first num components of this vector formatted as a string. More useful for printing due to the ability to limit overflow size
	 * @param num the number of components to format
	 * @return a string containing the first num components formatted as "(%n1,%n2,%n3)" or just "%n1" when dim==1 
	 */
	public String toString(int num){
		if(dim==0)
			return "";
		if(dim==1)
			return data.get(0)+"";
		StringBuilder retval=new StringBuilder("("+data.get(0));
		for(int adding=1;adding<num;adding++)
		{
			retval.append(",");
			retval.append(data.get(adding));
		}
		retval.append(")");
		return retval.toString();
	}
	public String toString(){
		if(dim==1)
			return data.get(0)+"";
		StringBuilder retval=new StringBuilder("("+data.get(0));
		for(int adding=1;adding<dim;adding++)
		{
			retval.append(",");
			retval.append(data.get(adding));
		}
		retval.append(")");
		return retval.toString();
	}
	@Override
	public nVector toVector() {
		return new nVector(this.getDataCopy());
	}
	@Override
	public double[] toDoubles() {
		double[] d = new double[dim];
		for(int i=0;i<dim;i++)
		{
			d[i]=data.get(i);
		}
		return d;
	}
	@Override
	public int vectorSize() {
		return dim;
	}
	@Override
	public nVector fromVector(nVector in) {
		return in;
	}
}