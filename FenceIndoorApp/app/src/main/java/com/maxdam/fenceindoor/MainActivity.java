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
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.Gson;
import com.maxdam.fenceindoor.common.CommonStuff;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.model.Area;
import com.maxdam.fenceindoor.service.ServicePredict;
import com.maxdam.fenceindoor.service.ServiceScanWifi;
import com.maxdam.fenceindoor.service.ServiceTraining;

public class MainActivity extends Activity {

	private Button scanningBtn, trainingBtn, predictBtn, settingsBtn;
	private TextView area;
	private SessionData sessionData;
    private SharedPreferences prefs;

    @Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		//ottiene la sessione
        sessionData = SessionData.getInstance(this);

        //ottiene le preferenze
        prefs = PreferenceManager.getDefaultSharedPreferences(this);

        scanningBtn = (Button)findViewById(R.id.scanning);
        trainingBtn = (Button)findViewById(R.id.training);
        predictBtn = (Button)findViewById(R.id.predict);
        settingsBtn = (Button)findViewById(R.id.settings);

		area = (TextView)findViewById(R.id.area);

        scanningBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {
            Intent intent = new Intent(MainActivity.this, SceltaAreaActivity.class);
            startActivity(intent);
            }
        });

        trainingBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {
                //avvia il training in base ai dati gia' trasmessi
                Intent serviceTrainingIntent = new Intent(MainActivity.this, ServiceTraining.class);
                MainActivity.this.startService(serviceTrainingIntent);
            }
        });

		predictBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {

                //ottiene il min scan count
                int minScanCount;
                try {
                    minScanCount = Integer.parseInt(prefs.getString("min_scan_count", "3"));
                } catch(Exception e) {
                    minScanCount = 3;
                }

                //ottiene lo scan count
                int maxScanCount = 1;

                //richiama il servizio di scansione one-shot
                Intent serviceScanWifiIntent = new Intent(MainActivity.this, ServiceScanWifi.class);
                serviceScanWifiIntent.putExtra("minScanCount", minScanCount);
                serviceScanWifiIntent.putExtra("maxScanCount", maxScanCount);
                startService(serviceScanWifiIntent);
            }
        });

        settingsBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
                startActivity(intent);
            }
        });

        //receiver di training update
        IntentFilter filterTraining = new IntentFilter();
        filterTraining.addAction(ServiceTraining.TRAINING_UPDATED);
        registerReceiver(receiverTraining, filterTraining);

        //receiver di predict update
        IntentFilter filterPredict = new IntentFilter();
        filterPredict.addAction(ServicePredict.PREDICT_UPDATED);
        registerReceiver(receiverPredict, filterPredict);

        //receiver di wifiscan update
        IntentFilter filterWifiScanUpdate = new IntentFilter();
        filterWifiScanUpdate.addAction(ServiceScanWifi.WIFISCAN_UPDATE);
        registerReceiver(receiverWifiScanUpdate, filterWifiScanUpdate);
    }

    //evento di update training
    private BroadcastReceiver receiverTraining = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            Toast.makeText(MainActivity.this, "Training effettuato con successo", Toast.LENGTH_SHORT).show();
        }
    };

    //evento di update predict
    private BroadcastReceiver receiverPredict = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            //ottiene l'area predict
            Area areaPredict = new Gson().fromJson(intent.getStringExtra("areaPredict"), Area.class);

            Toast.makeText(MainActivity.this, "Ti trovi nell'area: " + areaPredict.getArea(), Toast.LENGTH_LONG).show();
        }
    };

    //evento di fine scansione
    private BroadcastReceiver receiverWifiScanUpdate = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            //chiama una predict sul server inviandogli la scansione appena fatta
            String wifiScanListJson = intent.getStringExtra("wifiScanList");
            Intent servicePredictIntent = new Intent(MainActivity.this, ServicePredict.class);
            servicePredictIntent.putExtra("scan", wifiScanListJson);
            MainActivity.this.startService(servicePredictIntent);
        }
    };


    @Override
    protected void onResume() {
        super.onResume();

        //receiver di training update
        IntentFilter filterTraining = new IntentFilter();
        filterTraining.addAction(ServiceTraining.TRAINING_UPDATED);
        registerReceiver(receiverTraining, filterTraining);

        //receiver di predict update
        IntentFilter filterPredict = new IntentFilter();
        filterPredict.addAction(ServicePredict.PREDICT_UPDATED);
        registerReceiver(receiverPredict, filterPredict);

        //receiver di wifiscan update
        IntentFilter filterWifiScanUpdate = new IntentFilter();
        filterWifiScanUpdate.addAction(ServiceScanWifi.WIFISCAN_UPDATE);
        registerReceiver(receiverWifiScanUpdate, filterWifiScanUpdate);
    }

    @Override
    protected void onPause() {
        //unregister dei receivers
        unregisterReceiver(receiverTraining);
        unregisterReceiver(receiverPredict);
        unregisterReceiver(receiverWifiScanUpdate);

        super.onPause();
    }

	@Override
	protected void onStart() {
		super.onStart();

        //se siamo in debug cambia il serer nel server di debug
        if(prefs.getBoolean("debug", false)) {
            SharedPreferences.Editor editor = prefs.edit();
            editor.putString("server_path", CommonStuff.DEBUG_SERVER_PATH);
            editor.commit();
        }
	}

    @Override
    public void onBackPressed() {
        super.onBackPressed();
        startActivity(new Intent(this, MainActivity.class));
        finish();
    }
}
