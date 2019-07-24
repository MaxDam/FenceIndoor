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
import com.google.gson.reflect.TypeToken;
import com.maxdam.fenceindoor.adapter.ListaWifiAdapter;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.model.WifiScan;
import com.maxdam.fenceindoor.service.ServiceScanWifi;
import com.maxdam.fenceindoor.service.ServiceSendData;

import java.lang.reflect.Type;
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

            //ottiene la wifiScanList dall'intent
            Type listType = new TypeToken<List<WifiScan>>(){}.getType();
            List<WifiScan> wifiScanList = new Gson().fromJson(intent.getStringExtra("wifiScanList"), listType);

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

            //invia i dati della singola scansione al server
            /*Intent serviceSendDataIntent = new Intent(getApplicationContext(), ServiceSendData.class);
            Gson gson = new GsonBuilder().excludeFieldsWithModifiers(Modifier.TRANSIENT).create();
            String scanJson = gson.toJson(wifiScanList.toArray(new WifiScan[wifiScanList.size()]), WifiScan[].class);
            serviceSendDataIntent.putExtra("area", area);
            serviceSendDataIntent.putExtra("scan", scanJson);
            startService(serviceSendDataIntent);*/
        }
    };

    //evento di update predict
    private BroadcastReceiver receiverWifiScanEnd = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {

            //invia i dati al server di tutte le scansioni effettuate
            Intent serviceSendDataIntent = new Intent(getApplicationContext(), ServiceSendData.class);
            String scansJson = intent.getStringExtra("scans");
            serviceSendDataIntent.putExtra("area", area);
            serviceSendDataIntent.putExtra("scans", scansJson);
            startService(serviceSendDataIntent);

            //torna alla main activity
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

        //ottiene il numero di scansioni da skippare
        int skipScanCount;
        try {
            skipScanCount = Integer.parseInt(prefs.getString("skip_scan_count", "3"));
        } catch(Exception e) {
            skipScanCount = 3;
        }

        //ottiene il numero di scansioni da fare in fase di train
        int maxScanCount;
        try {
            maxScanCount = Integer.parseInt(prefs.getString("train_scan_count", "10"));
        } catch(Exception e) {
            maxScanCount = 10;
        }

        //richiama il servizio di scansione wifi
        Intent serviceScanWifiIntent = new Intent(this, ServiceScanWifi.class);
        serviceScanWifiIntent.putExtra("skipScanCount", skipScanCount);
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
