package com.maxdam.fenceindoor.util;

import java.util.Collection;
import java.util.Iterator;

public class StringUtil {

	//effettua il join delle stringhe
	public static String join(Collection<?> col, String delim) {
	    StringBuilder sb = new StringBuilder();
	    Iterator<?> iter = col.iterator();
	    if (iter.hasNext())
	        sb.append(iter.next().toString());
	    while (iter.hasNext()) {
	        sb.append(delim);
	        sb.append(iter.next().toString());
	    }
	    return sb.toString();
	}
	
	//rimuove l'ultimo carattere di una stringa
	public static String removeLastChar(String str) {
        return str.substring(0,str.length()-1);
    }
}
