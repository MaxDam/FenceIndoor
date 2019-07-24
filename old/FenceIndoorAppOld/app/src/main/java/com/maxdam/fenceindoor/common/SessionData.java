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

}
