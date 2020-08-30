import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.JFrame;

import grf.Slider;
import neuralNet.AdaptiveRateNNTrainer;
import neuralNet.Encoder;
import neuralNet.NeuralNet;
import neuralNet.NeuralNetHelper;
import neuralNet.NeuralNetworkTrainer;
import neuralNet.WeightLayer;
import neuralNet.nVector;
import no.uib.cipr.matrix.Vector;

public class Interface {
	static final double scale_down = 2;
	static final int fix_x = (int) (256/scale_down);
	static final int fix_y = (int) (168/scale_down);

	static final int Bx=128;
	static final int By=128;
	static final int ix=128;
	static final int iy=128;

	//Data format variables
	static int[] internalLayers = new int[]{164,32,1024};
	final static int dimNum = 1;
	static int dim = internalLayers[dimNum];
	static final boolean useRandomRuns = true;
	final static String FILE_NAME_STRING = "ANIME";
	final static boolean AUTO_ENCODER = true;
	final static String AI_NAME = "AnimeBig.net";
	final static boolean MONOCHROME = false;
	final static boolean ACTUALLY_SAVE = true;
	final static boolean TRAIN = false;
	final static boolean USE_RELU = true;
	final static boolean EXPLORE = false;

	//programmatic variables (do not touch)
	static nVector[] vBig;
	static nVector[] vSmall;
	static Random rng = new Random();
	static volatile boolean coo = true;
	static NeuralNet syk;
	static WeightLayer svd = null;
	
	//UI variables
	final static int SLIDER_X = 250;
	final static int SLIDER_W = 20;
	final static int SLIDER_H = 120;
	final static int SLIDER_Y = 20;
	final static int SLIDER_STRIDE = 26;
	final static double sliderNum = .1;
	static Slider[] components = new Slider[dim];
	static final int sampleNum = 30;
	
	//hyperparameters
	final static int runs = 1;
	final static int training = 2;
	final static double DIP_RATE = .93;
	static double ep = 0.001;
	final static double THRESHOLD = USE_RELU?.02:0;
	static int countt = 0;
	final static int MAX_EXAMPLES = 2000;
	
	
	
	public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException
	{
		int cc;
		NeuralNet temp;
		if(TRAIN)
		{
			try{
				temp = NeuralNet.fromFile(new File(AI_NAME));
				internalLayers = temp.getInternalLayers();
				cc = shuffleImages(rng);
			} catch(FileNotFoundException e)
			{
				cc = shuffleImages(rng);
				if(AUTO_ENCODER)
					temp = new NeuralNet(Bx*By*(MONOCHROME?1:3),internalLayers,Bx*By*(MONOCHROME?1:3),.02d,0, rng);
				else
					temp = new NeuralNet(cc,internalLayers,Bx*By*(MONOCHROME?1:3),.5d,0, rng);
				System.out.println("failed load, making new AI");
				System.out.println(e.getMessage());
			}
		} else {
			temp = NeuralNet.fromFile(new File(AI_NAME));
			internalLayers = temp.getInternalLayers();
			cc = temp.getIns();
			shuffleImages(rng);
			dim = internalLayers[dimNum];
			components = new Slider[dim];
			
		}
		for(int i=0;i<dim;i++)
		{
			components[i] = new Slider(SLIDER_X+SLIDER_STRIDE*i, SLIDER_Y, SLIDER_W, SLIDER_H);
			//principal component analysis sliders
		}
		final int ccc = cc;//# of training examples
		
		System.out.println(ccc);
		System.out.println(Arrays.toString(internalLayers));
		
		syk = temp;
		
		
		
		JFrame window = new JFrame();
		window.addWindowListener(new WindowListener(){

			public void windowClosing(WindowEvent arg0) {
				coo=false;
			}
			
			public void windowActivated(WindowEvent arg0) { } public void windowClosed(WindowEvent arg0) { } public void windowDeactivated(WindowEvent arg0) { } public void windowDeiconified(WindowEvent arg0) { } public void windowIconified(WindowEvent arg0) { } public void windowOpened(WindowEvent arg0) { }
			
		});
		window.setSize(Interface.Bx+250,Interface.By+170);
		window.addMouseMotionListener(new MouseMotionListener(){

			@Override
			public void mouseDragged(MouseEvent arg0) {
				for(int i=0;i<dim;i++)
				{
					components[i].MouseMove(arg0.getPoint());
				}
				window.repaint();
			}

			@Override
			public void mouseMoved(MouseEvent arg0) {
				
			}
			
		});
		//window.setExtendedState(JFrame.MAXIMIZED_BOTH);
		window.add(new JComponent(){

			private static final long serialVersionUID = 1L;
			@Override
			public void paintComponent(Graphics gg)
			{
				if(svd!=null)
				{
					nVector bob = nVector.getRandomVector(rng, dim, 0, 0);
					for(int i=0;i<dim;i++)
					{
						components[i].drawSelf(gg);
						bob.setComponent(i, components[i].sliderPos(-sliderNum, sliderNum));
					}
					
					bob = syk.getGAN(dimNum).decode(svd.applyWithBias(bob), true, USE_RELU);
					BufferedImage img = decodeImage(bob,Interface.Bx,By,MONOCHROME);
					gg.drawImage(img, 20, 20, null);
				}
				
			}
		});
		window.setVisible(true);
		window.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE );
		
		syk.setTrainingBatchSize(0);
		NeuralNetworkTrainer tr = new NeuralNetworkTrainer(syk, useRandomRuns, runs, training, ep, DIP_RATE, n->{}, THRESHOLD, rng);
		
		if(TRAIN)
			while(true)
			{
				//AUTO_ENCODER
				long time = System.currentTimeMillis();
				double ll;
				nVector values;
				if(AUTO_ENCODER)
				{
					svd = nVector.SVD(syk.samplesWeightLayer(vBig,USE_RELU,dimNum));
					values = svd.inverseMult(syk.getGAN(dimNum).encode(vBig[sampleNum], USE_RELU));
				}
				else
				{
					svd = nVector.SVD(syk.samplesWeightLayerOneHot(USE_RELU,dimNum));
					values = svd.inverseMult(syk.getGAN(dimNum).encode(new nVector(ccc,sampleNum), USE_RELU));
				}
				double[] vs = values.toDoubles();
				for(int i=0;i<dim;i++)
				{
					components[i].setValManually(-sliderNum, sliderNum, vs[i]);
				}
				window.repaint();
				if(coo)
				{
					if(AUTO_ENCODER)
						ll = tr.runFullLoopTraining(vBig, vBig, true, true);
					else
						ll = tr.runFullLoopTrainingOneHot(vBig, ccc, true, true);
					System.out.println(ll/runs/vBig.length);
					if(ACTUALLY_SAVE)
						syk.writeToFile(new File(AI_NAME));
					
				}
				if(!coo)
				{
					System.out.println("done");
					System.exit(0);
				}
				shuffleImages(rng);
			}
		else
		{
			int n=0;
			do
			{
				nVector values;
				if(AUTO_ENCODER)
				{
					svd = nVector.SVD(syk.samplesWeightLayer(vBig,USE_RELU,dimNum));
					values = svd.inverseMult(syk.getGAN(dimNum).encode(vBig[n%ccc], USE_RELU));
				}
				else
				{
					svd = nVector.SVD(syk.samplesWeightLayerOneHot(USE_RELU,dimNum));
					values = svd.inverseMult(syk.getGAN(dimNum).encode(new nVector(ccc,n%ccc), USE_RELU));
				}
				double[] vs = values.toDoubles();
				for(int i=0;i<dim;i++)
				{
					components[i].setValManually(-sliderNum, sliderNum, vs[i]);
				}
				window.repaint();
				n++;
				Thread.sleep(5000);
			} while(coo&&EXPLORE);
		}
		
	}
	private static int shuffleImages(Random rng) throws IOException
	{
		int ccc=MAX_EXAMPLES;
		int cc;
		File[] files;
		{
			File directory = new File(FILE_NAME_STRING);
			File[] allFiles = directory.listFiles(pathname -> pathname.getName().toLowerCase().matches(".*\\.(png|jpg|jpeg)$"));
			
			
			cc = allFiles.length<ccc?allFiles.length:ccc;
			files = new File[cc];
			for(int i=0;i<cc;i++)
			{
				int random = i+rng.nextInt(cc-i);
				File f = allFiles[random];
				allFiles[random] = allFiles[i];
				files[i] = f;
			}
		}
		
		
		System.out.println("Showing: "+files[sampleNum].getName());
		vBig = new nVector[cc];
		cc=0;
		for(File f:files)
		{
			if(cc>=ccc)
				break;
			Image vv = ImageIO.read(f);
			BufferedImage cp = new BufferedImage(Bx,By,BufferedImage.TYPE_INT_ARGB);
			cp.getGraphics().drawImage(vv,0,0,Bx,By,null);
			
			vBig[cc]=encodeImage(cp,MONOCHROME);
			cc++;
		}
		return cc;
	}
	private static nVector encodeImage(BufferedImage im, boolean monochrome)
	{
		int w=im.getWidth();
		int h=im.getHeight();
		double[] data = new double[w*h*(monochrome?1:3)];
		int ind=0;
		for(int y=0;y<h;y++)
		{
			for(int x=0;x<w;x++)
			{
				int rgb = im.getRGB(x, y);
				data[ind++] = ((rgb>> 16) & 0xFF)/255.0f;
				if(!monochrome)
				{
					data[ind++] = ((rgb>> 8) & 0xFF)/255.0f;
					data[ind++] = ((rgb) & 0xFF)/255.0f;
				}
			}
		}
		return new nVector(data);
	}
	private static BufferedImage decodeImage(nVector v, int w, int h, boolean monochrome)
	{
		BufferedImage im = new BufferedImage(w,h, BufferedImage.TYPE_INT_ARGB);
		if(monochrome)
			im = new BufferedImage(w,h, BufferedImage.TYPE_BYTE_GRAY);
		Graphics g =im.getGraphics();
		Vector data = v.getDataCopy();
		int u=0;
		for(int y=0;y<h;y++)
		{
			for(int x=0;x<w;x++)
			{
				int r=(int)(255*data.get(u++));
				int gg;
				int b;
				if(!monochrome)
				{
					gg=(int)(255*data.get(u++));
					b=(int)(255*data.get(u++));
				} else {
					gg=r;
					b=r;
				}
				r=Math.max(Math.min(255, r), 0);
				gg=Math.max(Math.min(255, gg), 0);
				b=Math.max(Math.min(255, b), 0);
				g.setColor(new Color(r,gg,b));
				g.drawLine(x, y, x, y);
			}
		}
		return im;
	}
}
