package com.maxdam.fenceindoor.service;

import android.app.IntentService;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.RemoteException;
import android.preference.PreferenceManager;

import com.maxdam.fenceindoor.model.WifiScan;
import com.maxdam.fenceindoor.util.BluetoothUtils;

import org.altbeacon.beacon.Beacon;
import org.altbeacon.beacon.BeaconConsumer;
import org.altbeacon.beacon.BeaconManager;
import org.altbeacon.beacon.BeaconParser;
import org.altbeacon.beacon.RangeNotifier;
import org.altbeacon.beacon.Region;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class ServiceScanIBeacon extends IntentService implements BeaconConsumer {

	private String TAG = ServiceScanIBeacon.class.getName();

    public static final String IBEACONSCAN_UPDATE = "broadcast.IBEACONSCAN_UPDATE";
    public static final String IBEACONSCAN_END = "broadcast.IBEACONSCAN_END";

	private BeaconManager beaconManager;

	//semaforo che indica la scansione bloccata
	public AtomicBoolean activeScan = new AtomicBoolean(false);

	private SharedPreferences prefs;

	private final Handler handler = new Handler();

    private int scanCount = 0;

    private int skipScanCount = 0;
    private int maxScanCount = 1;

	private int wifiLevelScale = 100;

	private List<List<WifiScan>> scans;

	public ServiceScanIBeacon() {
		super("ServiceScanIBeacon");
    }

	@Override
	protected void onHandleIntent(Intent intent) {

		//ottiene le preferenze
		prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

        activeScan.set(true);
        scanCount = 0;

        //preleva i parametri di ingresso
        skipScanCount = intent.getExtras().getInt("skipScanCount", 0);
        maxScanCount = intent.getExtras().getInt("maxScanCount", 10);
        maxScanCount += skipScanCount;

        //start del beacon detect
        startBeaconDetect();
	}

	public void doInBackground()
	{
		handler.postDelayed(new Runnable() {

			@Override
			public void run()
			{
                //effettua il bind del servizio
                beaconManager.bind(ServiceScanIBeacon.this);
			}
		}, 1000);
	}

    //start beacon detect
    public void startBeaconDetect() {
        //attiva il bluetooth
        BluetoothUtils.setBluetooth(true);

        //inizializza il beacon scanner
        beaconManager = BeaconManager.getInstanceForApplication(this);
        beaconManager.getBeaconParsers().add(new BeaconParser()
                .setBeaconLayout("m:2-3=0215,i:4-19,i:20-21,i:22-23,p:24-24"));

        //effettua il bind
        beaconManager.bind(this);
    }

    //stop beacon detect
    public void stopBeaconDetect() {
        //effettua l'unbind
        beaconManager.unbind(this);

        //disattiva il bluetooth
        BluetoothUtils.setBluetooth(false);
    }

    @Override
    public void onBeaconServiceConnect() {
        beaconManager.addRangeNotifier(new RangeNotifier() {
            @Override
            public void didRangeBeaconsInRegion(Collection<Beacon> beacons, Region region) {

                //se la scansione non e' attiva, esce
                if (!activeScan.get()) return;

                //Log.i(TAG, "trovati " + beacons.size() + " beacons");
                if (beacons.size() > 0) {

                    //scorre i dispositivi trovati
                    List<Beacon> beaconList = new ArrayList<Beacon>(beacons);
                    for(final Beacon beacon : beacons) {

                        //single beacon detected
                        String uuid = beacon.getId1()+"-"+beacon.getId2()+"-"+beacon.getId3();
                        int rssi = beacon.getRssi();
                        int txpower = beacon.getTxPower();
                        double distance = Beacon.getDistanceCalculator().calculateDistance(txpower, rssi);

                        beaconList.add(beacon);
                    }
                }

                if(scanCount >= skipScanCount) {

                    //salva la wifiScanList nella lista delle scansioni
                    //scans.add(wifiScanList);

                    //invia il broadcast per notificare l'aggiornamento della lista beacons
                    Intent broadcastIntentUpdate = new Intent();
                    broadcastIntentUpdate.setAction(IBEACONSCAN_UPDATE);
                    //Gson gson = new GsonBuilder().excludeFieldsWithModifiers(Modifier.TRANSIENT).create();
                    //Type listType = new TypeToken<List<WifiScan>>(){}.getType();
                    //broadcastIntentUpdate.putExtra("wifiScanList", gson.toJson(wifiScanList, listType));
                    sendBroadcast(broadcastIntentUpdate);
                }

                //effettua l' unbind del servizio
                beaconManager.unbind(ServiceScanIBeacon.this);

                //si richiama ricorsivamente
                doInBackground();

                //incrementa le scansioni effettuate
                scanCount++;

                //se ha raggiunto il numero massimo di scansioni.. si ferma
                if(scanCount >= maxScanCount) {

                    //invia il broadcast per notificare la fine del ciclo di scansione
                    Intent broadcastIntentEnd = new Intent();
                    broadcastIntentEnd.setAction(IBEACONSCAN_END);
                    //Type listType = new TypeToken<List<List<WifiScan>>>(){}.getType();
                    //Gson gson = new GsonBuilder().excludeFieldsWithModifiers(Modifier.TRANSIENT).create();
                    //broadcastIntentEnd.putExtra("scans", gson.toJson(scans, listType));
                    sendBroadcast(broadcastIntentEnd);

                    //stop della scansione ed azzeramento del count
                    activeScan.set(false);
                    scanCount = 0;
                    stopBeaconDetect();
                }
            }
        });

        try {
            beaconManager.startRangingBeaconsInRegion(new Region("myRangingUniqueId", null, null, null));
        } catch (RemoteException e) {

        }
    }
}
