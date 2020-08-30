package neuralNet;
import java.util.Random;

import no.uib.cipr.matrix.Vector;
/**
 * A helper class for certain types of classifier neural nets
 * indexOfmaxInRange is to be used to get the outputted classification given an output vector in double[] format
 * pickWeighted is for generative neural nets which output certain classes with probabilities
 * @author Zachary
 */
public class NeuralUtils {
	/**
	 * Picks the index of the largest double present in data in the range start->end-1. This will always scan end-start items
	 * @param start the index of the first element to scan
	 * @param end the index of the last element to scan + 1
	 * @param data the doubles to scan
	 * @return the index of the largest double in the data
	 */
	public static int indexOfMaxInRange(int start, int end, double[] data)
	{
		int maxx=start;
		for(int i=start;i<end;i++)
		{
			if(data[i]>data[maxx])
				maxx=i;
		}
		return maxx;
	}
	/**
	 * picks an index from start to end-1 randomly, weighted by the doubles present in data
	 * @param start the index to start
	 * @param end the index to end
	 * @param data a vector from which to pick the weights
	 * @param rng a random number generator
	 * @return a random index weighted by the doubles in data
	 */
	public static int pickWeighted(int start, int end, Vector data, Random rng)
	{
		//System.out.println(Arrays.toString(data));
		Vector dupes = data.copy();
		double totalSum=0;
		for(int a=start;a<end;a++)
		{
			totalSum+=data.get(a);
		}
		//divide chances by a.
		dupes.scale(1.0/totalSum);
		double check = rng.nextDouble();
		totalSum=0;
		for(int a=start;a<end;a++)
		{
			if(totalSum+dupes.get(a)>check)
				return a;
			totalSum+=dupes.get(a);
		}
		return end-1;
	}
}
