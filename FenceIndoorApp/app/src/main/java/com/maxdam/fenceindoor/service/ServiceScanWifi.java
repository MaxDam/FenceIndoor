package com.maxdam.fenceindoor.service;

import android.app.IntentService;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.os.Handler;
import android.preference.PreferenceManager;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.maxdam.fenceindoor.model.WifiScan;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class ServiceScanWifi extends IntentService {

	private String TAG = ServiceScanWifi.class.getName();

    public static final String WIFISCAN_UPDATE = "broadcast.WIFISCAN_UPDATE";
    public static final String WIFISCAN_END = "broadcast.WIFISCAN_END";

	//semaforo che indica la scansione bloccata
	public AtomicBoolean activeScan = new AtomicBoolean(false);

	private SharedPreferences prefs;

	private WifiManager mainWifi;
	private WifiReceiver receiverWifi;

	private final Handler handler = new Handler();

    private int scanCount = 0;
    private int maxScanCount = 1;

	public ServiceScanWifi() {
		super("ServiceScanWifi");
    }

	@Override
	protected void onHandleIntent(Intent intent) {

		//ottiene le preferenze
		prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

        activeScan.set(true);
        scanCount = 0;

        maxScanCount = intent.getExtras().getInt("scanCount");

        mainWifi = (WifiManager) getSystemService(Context.WIFI_SERVICE);

		if(mainWifi.isWifiEnabled()==false) {
			mainWifi.setWifiEnabled(true);
		}

		doInBackground();
	}

	public void doInBackground()
	{
		handler.postDelayed(new Runnable() {

			@Override
			public void run()
			{
				mainWifi = (WifiManager) getSystemService(Context.WIFI_SERVICE);

				receiverWifi = new WifiReceiver();
				registerReceiver(new WifiReceiver(), new IntentFilter(
						WifiManager.SCAN_RESULTS_AVAILABLE_ACTION));
				mainWifi.startScan();
			}
		}, 1000);

	}

	//receiver della scansione wifi
	class WifiReceiver extends BroadcastReceiver {
		public void onReceive(Context c, Intent intent) {

            //se la scansione non e' attiva, esce
            if (!activeScan.get()) return;

			//ottiene la lista delle wifi scansionate
			final List<WifiScan> wifiScanList = new ArrayList<WifiScan>();
			List<ScanResult> scanResultList = mainWifi.getScanResults();
			for (int i = 0; i < scanResultList.size(); i++) {

				ScanResult scanResult = scanResultList.get(i);

				WifiScan wifiScan = new WifiScan();
				wifiScan.setWifiName(scanResult.SSID);
				wifiScan.setWifiLevel(WifiManager.calculateSignalLevel(scanResult.level, 100));

				wifiScanList.add(wifiScan);
			}

            //invia il broadcast per notificare l'aggiornamento della lista wifi
            Intent broadcastIntentUpdate = new Intent();
            broadcastIntentUpdate.setAction(WIFISCAN_UPDATE);
            Gson gson = new GsonBuilder().excludeFieldsWithModifiers(Modifier.TRANSIENT).create();
            broadcastIntentUpdate.putExtra("wifiScanList", gson.toJson(wifiScanList.toArray(new WifiScan[wifiScanList.size()]), WifiScan[].class));
            sendBroadcast(broadcastIntentUpdate);

            //si autorichiama ricorsivamente
			doInBackground();

			//se ha raggiunto il numero massimo di scansioni.. si ferma
			scanCount++;
			if(scanCount >= maxScanCount) {

                //invia il broadcast per notificare la fine del ciclo di scansione
                Intent broadcastIntentEnd = new Intent();
                broadcastIntentEnd.setAction(WIFISCAN_END);
                sendBroadcast(broadcastIntentEnd);

                //stop della scansione ed azzeramento del count
                activeScan.set(false);
                scanCount = 0;
            }
		}
	}
}
