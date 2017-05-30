package com.maxdam.fenceindoor;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.text.InputType;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;

import com.google.gson.Gson;
import com.maxdam.fenceindoor.adapter.ListaAreeAdapter;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.model.Area;
import com.maxdam.fenceindoor.service.ServiceAddArea;
import com.maxdam.fenceindoor.service.ServiceAree;


import java.util.Arrays;
import java.util.List;

public class SceltaAreaActivity extends Activity {

	private ListView listView;
    private SessionData sessionData;
    private SharedPreferences prefs;
    private Button newAreaBtn;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_scelta_area);
        
		//ottiene il listview 
		listView = ((ListView)findViewById(R.id.aziendeListView));

        //ottiene la sessione
        sessionData = SessionData.getInstance(this);

        //ottiene le preferenze
        prefs = PreferenceManager.getDefaultSharedPreferences(this);

        //configura l'evento per il bottone new area
        newAreaBtn = (Button)findViewById(R.id.newAreaBtn);
        newAreaBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AlertDialog.Builder alert = new AlertDialog.Builder(SceltaAreaActivity.this);
                alert.setTitle("inserisci area");
                final EditText input = new EditText(SceltaAreaActivity.this);
                input.setInputType(InputType.TYPE_CLASS_TEXT);
                alert.setView(input);
                alert.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {

                        //crea l'oggetto area da inserire
                        Area newArea = new Area();
                        newArea.setArea(input.getText().toString());

                        //comunica con il server per l'inserimento dell'area
                        Intent serviceAddAreaIntent = new Intent(SceltaAreaActivity.this, ServiceAddArea.class);
                        serviceAddAreaIntent.putExtra("area", new Gson().toJson(newArea, Area.class));
                        SceltaAreaActivity.this.startService(serviceAddAreaIntent);
                    }
                });
                alert.show();
            }
        });

        //ottiene dal server la lista Aree
        Intent serviceAreeIntent = new Intent(this, ServiceAree.class);
        this.startService(serviceAreeIntent);
	}

    //evento di update acquisizione lista aree
    private BroadcastReceiver receiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {

            //ottiene dall'intent la lista aree restituita
            List<Area> listaAree = Arrays.asList(new Gson().fromJson(intent.getStringExtra("aree"), Area[].class));

            //aggiorna l'adapter
            ListaAreeAdapter adapter = new ListaAreeAdapter(SceltaAreaActivity.this, R.layout.scelta_area_item, listaAree);
            listView.setAdapter(adapter);

            //rende invisibile l'immagine di caricamento
            findViewById(R.id.loadImage).setVisibility(View.GONE);
        }
    };

    @Override
    protected void onResume() {
        IntentFilter filter = new IntentFilter();
        filter.addAction(ServiceAree.AREE_UPDATED);
        registerReceiver(receiver, filter);
        super.onResume();
    }

    @Override
    protected void onPause() {
        unregisterReceiver(receiver);
        super.onPause();
    }

}
