package com.maxdam.fenceindoor.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import com.maxdam.fenceindoor.R;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.model.WifiScan;
import com.maxdam.fenceindoor.util.RandomUtil;

import java.util.List;

public class ListaWifiAdapter extends ArrayAdapter<WifiScan> {

	private SessionData sessionData;
    private List<WifiScan> wifiScanList;

    public ListaWifiAdapter(Context context, int textViewResourceId, List<WifiScan> wifiScanList) {
        super(context, textViewResourceId, wifiScanList);

        this.wifiScanList = wifiScanList;

        //ottiene la sessione
        sessionData = SessionData.getInstance(this.getContext());
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
    	 ViewHolder viewHolder = null;
         if (convertView == null) {
             LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
             convertView = inflater.inflate(R.layout.scan_item, null);
             
             viewHolder = new ViewHolder();
             viewHolder.icon = (ImageView)convertView.findViewById(R.id.wifiImage);
             viewHolder.detail = (TextView)convertView.findViewById(R.id.nameWifi);
             viewHolder.subDetail = (TextView)convertView.findViewById(R.id.levelWifi);
             convertView.setTag(viewHolder);
         } else {
             viewHolder = (ViewHolder) convertView.getTag();
         }
         final WifiScan wifi = getItem(position);
         wifi.setColor(RandomUtil.randomDefaultColor());
         
         //set del nome e del livello di segnale
         viewHolder.detail.setText(wifi.getWifiName());
         viewHolder.subDetail.setText(wifi.getWifiLevel().toString());

         //colora lo sfondo dell'icona
         viewHolder.icon.setBackgroundColor(wifi.getColor());
         
         return convertView;
    }

    public void updateData(List<WifiScan> wifiScanList) {
        this.wifiScanList.clear();
        this.wifiScanList.addAll(wifiScanList);
        this.notifyDataSetChanged();
    }

	private class ViewHolder {
        public ImageView icon;
        public TextView detail;
        public TextView subDetail;
    }
}
