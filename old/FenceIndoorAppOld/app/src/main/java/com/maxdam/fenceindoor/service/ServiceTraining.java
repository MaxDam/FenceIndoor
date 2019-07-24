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

public class ServiceTraining extends IntentService {

	private String TAG = ServiceTraining.class.getName();

	public static final String TRAINING_UPDATED = "broadcast.TRAINING_UPDATED";

	private SharedPreferences prefs;

	//semaforo che indica esserci una sincronizzazione in atto
	public static final AtomicBoolean inProgress = new AtomicBoolean(false);

	public ServiceTraining() {
        super("ServiceTraining");
    }

	@Override
	protected void onHandleIntent(Intent intent) {

		//ottiene le preferenze
		prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

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
                String method = "training";

                //chiama il server
				JsonRestClient jsonClient = JsonRestClient
						.newInstance(prefs.getString("server_path", CommonStuff.DEFAULTSERVER_PATH))
	            		.addPath(method);
	            Log.d(TAG, "request: "+jsonClient);
	            String out = jsonClient.get().getOutputString();
	            Log.d(TAG, "response: "+out);
	            
		    	//invia il broadcast per notificare l'aggiornamento
		        Intent broadcastIntent = new Intent();
		        broadcastIntent.setAction(TRAINING_UPDATED);
		        this.sendBroadcast(broadcastIntent); 
	            
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
