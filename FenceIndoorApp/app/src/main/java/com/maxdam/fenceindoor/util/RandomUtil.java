package com.maxdam.fenceindoor.util;

import android.graphics.Color;

import java.util.Random;

public class RandomUtil {

	//torna un numero random da un minimo ad un massimo
	public static int randomInt(int min, int max) {
	    Random rand = new Random();
	    int randomNum = rand.nextInt((max - min) + 1) + min;
	    return randomNum;
	}
	
	//torna un qualunque colore random
	public static int randomColor() {
		return Color.argb(255, randomInt(0, 256), randomInt(0, 256), randomInt(0, 256));
	}
	
	//torna un colore random fra quelli di default
	/*public static int randomDefaultColor() {
		int[] colors= { Color.BLUE,	  Color.CYAN,	
						Color.DKGRAY, Color.GRAY,	
						Color.GREEN,  Color.MAGENTA,
						Color.RED,	  Color.YELLOW };
		Random rnd = new Random(); 
		return colors[rnd.nextInt(colors.length - 1)];
	}*/
	
	public static int randomDefaultColor() {

		String[] colors = {
			"F3B200", "77B900",	"2572EB",		
			"AD103C", "632F00", "B01E00",
			"C1004F", "7200AC", "4617B4",
			"006AC1", "008287", "199900",
			"00C13F", "FF981D", "FF2E12",
			"AA40FF", "1FAEFF", "56C5FF", 
			"91D100", "E1B700", "00A3A3", 
			"FE7C22"};
		
		Random rnd = new Random();
		String colorStr = colors[rnd.nextInt(colors.length - 1)];
		return Color.rgb(
	            Integer.valueOf( colorStr.substring( 0, 2 ), 16 ),
	            Integer.valueOf( colorStr.substring( 2, 4 ), 16 ),
	            Integer.valueOf( colorStr.substring( 4, 6 ), 16 ) );
	}
	
	//torna un colore random pastello
	public static int randomPastelColor() {
		
		int randomColor = randomColor();
		int redMix = (randomColor >> 16) & 0xFF;
		int greenMix = (randomColor >> 8) & 0xFF;
		int blueMix = (randomColor >> 0) & 0xFF;
		
		Random random = new Random();
	    int red = random.nextInt(256);
	    int green = random.nextInt(256);
	    int blue = random.nextInt(256);

	    // mix the color
        red = (red + redMix) / 2;
        green = (green + greenMix) / 2;
        blue = (blue + blueMix) / 2;
	    return Color.rgb(red, green, blue);
	}
}
