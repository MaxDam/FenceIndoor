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

    private int minScanCount = 0;
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

        //start del beacon detect
        startBeaconDetect();

        doInBackground();
	}

	public void doInBackground()
	{
		handler.postDelayed(new Runnable() {

			@Override
			public void run()
			{
				//...
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

                /*final List<Beacon> beaconList = new ArrayList<Beacon>(beacons);
                if(listView.getAdapter() == null) {
                    BeaconListAdapter adapter = new BeaconListAdapter(BeaconsActivity.this, beaconList);
                    listView.setAdapter(adapter);
                }
                else {
                    ((BeaconListAdapter)listView.getAdapter()).refill(beaconList);
                }*/

                if (beacons.size() > 0) {

                    //scorre i dispositivi trovati
                    for(final Beacon beacon : beacons) {

                        //single beacon detected
                        String uuid = beacon.getId1()+"-"+beacon.getId2()+"-"+beacon.getId3();
                        int rssi = beacon.getRssi();
                        int txpower = beacon.getTxPower();
                        double distance = Beacon.getDistanceCalculator().calculateDistance(txpower, rssi);
                        //...
                    }
                }
            }
        });

        try {
            beaconManager.startRangingBeaconsInRegion(new Region("myRangingUniqueId", null, null, null));
        } catch (RemoteException e) {

        }
    }
}
