package com.maxdam.fenceindoor;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.view.View;
import android.widget.ListView;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.maxdam.fenceindoor.adapter.ListaWifiAdapter;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.model.WifiScan;
import com.maxdam.fenceindoor.service.ServiceScanWifi;
import com.maxdam.fenceindoor.service.ServiceSendData;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ScanActivity extends Activity {

    private ListView listView;
    private SessionData sessionData;
    private String area = "";
    private SharedPreferences prefs;
    private ListaWifiAdapter wifiScanAdapter;

    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scan);

        //ottiene il listview
        listView = ((ListView)findViewById(R.id.wifiListView));

        //ottiene la sessione
        sessionData = SessionData.getInstance(this);

        //ottiene le preferenze
        prefs = PreferenceManager.getDefaultSharedPreferences(this);

        //acquisisce l'identificativo dell'area da scansionare
        Intent intent = getIntent();
        area = intent.getStringExtra("area");

        IntentFilter filterWifiScanUpdate = new IntentFilter();
        filterWifiScanUpdate.addAction(ServiceScanWifi.WIFISCAN_UPDATE);
        registerReceiver(receiverWifiScanUpdate, filterWifiScanUpdate);

        IntentFilter filterWifiScanEnd = new IntentFilter();
        filterWifiScanEnd.addAction(ServiceScanWifi.WIFISCAN_END);
        registerReceiver(receiverWifiScanEnd, filterWifiScanEnd);
    }

    //evento di update training
    private BroadcastReceiver receiverWifiScanUpdate = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {

            //ottiene la nuova lista all'intent
            List<WifiScan> wifiScanList = new ArrayList<WifiScan>(
                    Arrays.asList(
                            new Gson().fromJson(
                                    intent.getStringExtra("wifiScanList"),
                                    WifiScan[].class
                            )
                    )
            );

            //aggiorna la lista mantenendo lo scroll
            if (listView.getAdapter() == null) {
                wifiScanAdapter = new ListaWifiAdapter(getApplicationContext(), R.layout.scan_item, wifiScanList);
                listView.setAdapter(wifiScanAdapter);
            }
            else {
                wifiScanAdapter.updateData(wifiScanList);
            }

            //rende invisibile l'immagine di caricamento
            findViewById(R.id.loadImage).setVisibility(View.GONE);

            //invia i dati della scansione al server
            Intent serviceSendDataIntent = new Intent(getApplicationContext(), ServiceSendData.class);
            Gson gson = new GsonBuilder().excludeFieldsWithModifiers(Modifier.TRANSIENT).create();
            String scanJson = gson.toJson(wifiScanList.toArray(new WifiScan[wifiScanList.size()]), WifiScan[].class);
            serviceSendDataIntent.putExtra("area", area);
            serviceSendDataIntent.putExtra("scan", scanJson);
            startService(serviceSendDataIntent);
        }
    };

    //evento di update predict
    private BroadcastReceiver receiverWifiScanEnd = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            Intent intentMain = new Intent(ScanActivity.this, MainActivity.class);
            startActivity(intentMain);
        }
    };

    @Override
    protected void onStart() {
        super.onStart();

        //acquisisce l'identificativo dell'area da scansionare
        Intent intent = getIntent();
        area = intent.getStringExtra("area");

        //ottiene il min scan count
        int minScanCount;
        try {
            minScanCount = Integer.parseInt(prefs.getString("min_scan_count", "3"));
        } catch(Exception e) {
            minScanCount = 3;
        }

        //ottiene il max scan count
        int maxScanCount;
        try {
            maxScanCount = Integer.parseInt(prefs.getString("scan_count", "10"));
        } catch(Exception e) {
            maxScanCount = 10;
        }

        //richiama il servizio di scansione
        Intent serviceScanWifiIntent = new Intent(this, ServiceScanWifi.class);
        serviceScanWifiIntent.putExtra("minScanCount", minScanCount);
        serviceScanWifiIntent.putExtra("maxScanCount", maxScanCount);
        this.startService(serviceScanWifiIntent);
    }

    @Override
    protected void onPause()
    {
        unregisterReceiver(receiverWifiScanUpdate);
        unregisterReceiver(receiverWifiScanEnd);
        super.onPause();
    }

    @Override
    protected void onResume()
    {
        IntentFilter filterWifiScanUpdate = new IntentFilter();
        filterWifiScanUpdate.addAction(ServiceScanWifi.WIFISCAN_UPDATE);
        registerReceiver(receiverWifiScanUpdate, filterWifiScanUpdate);

        IntentFilter filterWifiScanEnd = new IntentFilter();
        filterWifiScanEnd.addAction(ServiceScanWifi.WIFISCAN_END);
        registerReceiver(receiverWifiScanEnd, filterWifiScanEnd);

        super.onResume();
    }
}
