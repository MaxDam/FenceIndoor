package com.maxdam.fenceindoor.common;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.SharedPreferences.Editor;

import com.google.gson.Gson;

public class SessionData {
	
	private static final String SESSION_DATA_SHARED_PREFS = SessionData.class.getSimpleName();
	private SharedPreferences sharedPrefs;
    private Editor prefsEditor;
	private Gson gson;
	 
	//ottiene l'istanza dell'oggetto
	public static SessionData getInstance(Context ctx) {

		return new SessionData(ctx);
	}
	
	//costruttore privato
	private SessionData(Context ctx) {
		this.sharedPrefs = ctx.getSharedPreferences(SESSION_DATA_SHARED_PREFS, Activity.MODE_PRIVATE);
        this.prefsEditor = sharedPrefs.edit();
		this.gson = new Gson();
	}


	/*private static String KEY_VETTOREMEZZO = "KEY_VETTOREMEZZO";
	public VettoreMezzo getVettoreMezzo() {
		 if(!sharedPrefs.contains(KEY_VETTOREMEZZO)) return null;
		 return gson.fromJson(sharedPrefs.getString(KEY_VETTOREMEZZO, ""), VettoreMezzo.class);
	}
	public void setVettoreMezzo(VettoreMezzo vettoreMezzo) {
		 prefsEditor.putString(KEY_VETTOREMEZZO, gson.toJson(vettoreMezzo)).commit();
	}

	private static String KEY_LISTAAREE = "KEY_LISTAAREE";
	public List<Area> getListaAree() {
		if(!sharedPrefs.contains(KEY_LISTAAREE)) return new ArrayList<Area>();
		return new ArrayList<Area>(Arrays.asList(new Gson().fromJson(sharedPrefs.getString(KEY_LISTAAREE, ""), Area[].class)));
	}
	public void setListaAree(List<Area> list) {
		prefsEditor.putString(KEY_LISTAAREE, gson.toJson(list.toArray(new Area[list.size()]))).commit();
	}
	*/
}
