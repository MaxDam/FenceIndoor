package com.maxdam.fenceindoor;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.graphics.Color;
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
	private TextView predictArea;
	private SessionData sessionData;
    private SharedPreferences prefs;
    private boolean predictMode = false;
    private int predictScanCount = 1;
    private int skipScanCount = 0;

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

        predictArea = (TextView)findViewById(R.id.predictArea);

        //ottiene il numero di scansioni da skippare
        try {
            skipScanCount = Integer.parseInt(prefs.getString("skip_scan_count", "3"));
        } catch(Exception e) {
            skipScanCount = 3;
        }

        //ottiene il predict scan count
        try {
            predictScanCount = Integer.parseInt(prefs.getString("predict_scan_count", "1"));
        } catch(Exception e) {
            predictScanCount = 1;
        }

        //configura il bottone di scanning wifi
        scanningBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {

                //disabilita il predict mode
                disablePredictMode();

                //chiama l'activity di scelta area
                Intent intent = new Intent(MainActivity.this, SceltaAreaActivity.class);
                startActivity(intent);
            }
        });

        //configura il bottone di training
        trainingBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {

                //disabilita il predict mode
                disablePredictMode();

                //avvia il training in base ai dati gia' trasmessi
                Intent serviceTrainingIntent = new Intent(MainActivity.this, ServiceTraining.class);
                MainActivity.this.startService(serviceTrainingIntent);
            }
        });

        //configura il bottone di predict
		predictBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {

                //se predict mode e' attivo lo disabilita, altrimenti lo abilita
                if(predictMode) {
                    //disabilita il predict mode
                    disablePredictMode();
                } else {
                    enablePredictMode();
                }

                //se siamo in predict mode..
                if(predictMode) {
                    //richiama il servizio di scansione one-shot
                    Intent serviceScanWifiIntent = new Intent(MainActivity.this, ServiceScanWifi.class);
                    serviceScanWifiIntent.putExtra("skipScanCount", skipScanCount);
                    serviceScanWifiIntent.putExtra("maxScanCount", predictScanCount);
                    startService(serviceScanWifiIntent);
                } else {
                    predictArea.setText("");
                }
            }
        });

        //configura il bottone di setting
        settingsBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {

                //disabilita il predict mode
                disablePredictMode();

                //chiama l'activity di setting
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

        //receiver di wifiscan end
        IntentFilter filterWifiScanEnd = new IntentFilter();
        filterWifiScanEnd.addAction(ServiceScanWifi.WIFISCAN_END);
        registerReceiver(receiverWifiScanEnd, filterWifiScanEnd);

        //disabilita il predict mode
        disablePredictMode();
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

            //Toast.makeText(MainActivity.this, "Ti trovi in: " + areaPredict.getArea(), Toast.LENGTH_LONG).show();
            if(predictMode) {
                predictArea.setText("Ti trovi in: " + areaPredict.getArea());
            } else {
                predictArea.setText("");
            }
        }
    };

    //evento di scansione
    private BroadcastReceiver receiverWifiScanUpdate = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {

            //chiama una predict sul server inviandogli la scansione appena fatta
            /*String wifiScanListJson = intent.getStringExtra("wifiScanList");
            Intent servicePredictIntent = new Intent(MainActivity.this, ServicePredict.class);
            servicePredictIntent.putExtra("scan", wifiScanListJson);
            MainActivity.this.startService(servicePredictIntent);*/
        }
    };

    //evento di fine scansione
    private BroadcastReceiver receiverWifiScanEnd = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {

            //se siamo in predict mode..
            if(predictMode) {

                //chiama una predict sul server inviandogli la scansione appena fatta
                String wifiScans = intent.getStringExtra("scans");
                Intent servicePredictIntent = new Intent(MainActivity.this, ServicePredict.class);
                servicePredictIntent.putExtra("scans", wifiScans);
                MainActivity.this.startService(servicePredictIntent);

                //richiama il servizio di scansione one-shot
                Intent serviceScanWifiIntent = new Intent(MainActivity.this, ServiceScanWifi.class);
                serviceScanWifiIntent.putExtra("skipScanCount", skipScanCount);
                serviceScanWifiIntent.putExtra("maxScanCount", predictScanCount);
                startService(serviceScanWifiIntent);
            } else {
                predictArea.setText("");
            }
        }
    };

    //disabilita il predict mode
    private void disablePredictMode() {
        predictMode = false;
        predictArea.setText("");
        predictBtn.setBackgroundColor(Color.parseColor("#DA542C"));
    }

    //abilita il predict mode
    private void enablePredictMode() {
        predictMode = true;
        predictArea.setText("...");
        predictBtn.setBackgroundColor(Color.parseColor("#FF845C"));
    }

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

        //receiver di wifiscan end
        IntentFilter filterWifiScanEnd = new IntentFilter();
        filterWifiScanEnd.addAction(ServiceScanWifi.WIFISCAN_END);
        registerReceiver(receiverWifiScanEnd, filterWifiScanEnd);

        //disabilita il predict mode
        disablePredictMode();
    }

    @Override
    protected void onPause() {
        //unregister dei receivers
        unregisterReceiver(receiverTraining);
        unregisterReceiver(receiverPredict);
        unregisterReceiver(receiverWifiScanUpdate);
        unregisterReceiver(receiverWifiScanEnd);

        //disabilita il predict mode
        disablePredictMode();

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

        //ottiene il numero di scansioni da skippare
        try {
            skipScanCount = Integer.parseInt(prefs.getString("skip_scan_count", "3"));
        } catch(Exception e) {
            skipScanCount = 3;
        }

        //ottiene il numero di scansioni da effettuare in fase di predict
        try {
            predictScanCount = Integer.parseInt(prefs.getString("predict_scan_count", "1"));
        } catch(Exception e) {
            predictScanCount = 1;
        }

        //disabilita il predict mode
        disablePredictMode();
	}

    @Override
    public void onBackPressed() {
        super.onBackPressed();
        startActivity(new Intent(this, MainActivity.class));
        finish();

        //disabilita il predict mode
        disablePredictMode();
    }
}
