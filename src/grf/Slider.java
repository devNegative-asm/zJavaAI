package grf;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;

public class Slider {
	private double value = 0;
	private int x;
	private int y;
	private int width;
	private Rectangle me;
	private Rectangle zoom;
	private Rectangle slider;
	public Slider(int x, int y, int width, int height) {
		super();
		this.x = x;
		this.y = y;
		value = .5;
		this.width = width;
		me = new Rectangle(x,y,width,height);
		zoom = new Rectangle(x,y+width/2,width,height-width);
		slider = new Rectangle(x,y+height/2-width/2,width,width);
	}
	public void setValManually(double min, double max, double val)
	{
		value = (val-min)/(max-min);
		slider = new Rectangle(x,(int) (y+value*zoom.height),width,width);
	}
	public double sliderPos(double min, double max)
	{
		return min + value * (max-min);
	}
	public void MouseMove(Point p)
	{
		p = (Point) p.clone();
		p.translate(-8, -31);
		if(zoom.contains(p))
		{
			value = (p.y-zoom.y)/(double)zoom.height;
		}
		if(me.contains(p)&&p.y<zoom.y)
		{
			value = 0;
		}
		if(me.contains(p)&&p.y>zoom.y+zoom.height)
		{
			value = 1;
		}
		slider = new Rectangle(x,(int) (y+value*zoom.height),width,width);
	}
	public void drawSelf(Graphics g)
	{
		Graphics2D pic = (Graphics2D) g;
		pic.setColor(Color.GRAY);
		pic.fill(me);
		pic.setColor(Color.RED);
		pic.fill(slider);
		
	}
}
