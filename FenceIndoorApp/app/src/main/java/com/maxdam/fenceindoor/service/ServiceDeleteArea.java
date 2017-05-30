package com.maxdam.fenceindoor.service;

import android.app.IntentService;
import android.content.Intent;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.util.Log;

import com.maxdam.fenceindoor.common.CommonStuff;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.restclient.JsonRestClient;

import java.util.concurrent.atomic.AtomicBoolean;

public class ServiceDeleteArea extends IntentService {

	private String TAG = ServiceDeleteArea.class.getName();

    private SharedPreferences prefs;

	//semaforo che indica esserci una sincronizzazione in atto
	public static final AtomicBoolean inProgress = new AtomicBoolean(false);

	public ServiceDeleteArea() {
        super("ServiceDeleteArea");
    }

	@Override
	protected void onHandleIntent(Intent intent) {

		//ottiene le preferenze
		prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

		//preleva l'id area da cancellare
		String area = intent.getExtras().getString("area");

		//se e' gia attivo un tracing.. esce
		if (inProgress.get()) return;

		try {
			//imposta ad on il semaforo
			inProgress.set(true);
	
			//ottiene la sessione
			SessionData sessionData = SessionData.getInstance(this);
			
			//chiama il server per aggiornare le tipologie di lettera
		    try {
                //acquisisce il metodo da invocare
                String method = "deleteArea";

                //chiama il server
				JsonRestClient jsonClient = JsonRestClient
						.newInstance(prefs.getString("server_path", CommonStuff.DEFAULTSERVER_PATH))
	            		.addPath(method)
						.addParam(area);

				Log.d(TAG, "request: "+jsonClient);
				String out = jsonClient.get().getOutputString();
				Log.d(TAG, "response: "+out);

                //aggiorla la lista aree chiamando il servizio
                Intent serviceAreeIntent = new Intent(this, ServiceAree.class);
                this.startService(serviceAreeIntent);
	            
		    } catch (Exception e) {
		    	Log.e(TAG, "errore nella comunicazione con il server", e);
		    }
		}
		finally {
			//reimposta a off il semaforo
			inProgress.set(false);
		}
	}
}
