package neuralNet;
public interface Vectorizable<X> {
	public nVector toVector();
	public double[] toDoubles();
	public int vectorSize();
	public X fromVector(nVector in);
}
