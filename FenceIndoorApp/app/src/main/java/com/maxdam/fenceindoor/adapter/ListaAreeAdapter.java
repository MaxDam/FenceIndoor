package com.maxdam.fenceindoor.adapter;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import com.maxdam.fenceindoor.R;
import com.maxdam.fenceindoor.ScanActivity;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.model.Area;
import com.maxdam.fenceindoor.util.RandomUtil;

import java.util.List;

public class ListaAreeAdapter extends ArrayAdapter<Area> {

	private SessionData sessionData;

    public ListaAreeAdapter(Context context, int textViewResourceId, List<Area> objects) {
        super(context, textViewResourceId, objects);
        
      //ottiene la sessione
        sessionData = SessionData.getInstance(this.getContext());
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
    	 ViewHolder viewHolder = null;
         if (convertView == null) {
             LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
             convertView = inflater.inflate(R.layout.scelta_area_item, null);
             
             viewHolder = new ViewHolder();
             viewHolder.icon = (ImageView)convertView.findViewById(R.id.areaImage);
             viewHolder.detail = (TextView)convertView.findViewById(R.id.nameArea);
             viewHolder.subDetail = (TextView)convertView.findViewById(R.id.scanCount);
             convertView.setTag(viewHolder);
         } else {
             viewHolder = (ViewHolder) convertView.getTag();
         }
         final Area area = getItem(position);
         area.setColor(RandomUtil.randomDefaultColor());
         
         //set del nome e dello scan count
         viewHolder.detail.setText(area.getName());
         viewHolder.subDetail.setText(area.getScanCount().toString());
         
         //onclick sull'item
         convertView.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				onClickListItem(area);
			}
        });
         
        //colora lo sfondo dell'icona
        viewHolder.icon.setBackgroundColor(area.getColor());
         
        return convertView;
    }

    //click dell'elemento
    private void onClickListItem(Area area) {
    	//ottiene il context
		final Context ctx = ListaAreeAdapter.this.getContext();

		//richiama l'activity ListaAree passando l'area selezionata
        Intent intent = new Intent(ctx, ScanActivity.class);
        intent.putExtra("area", area.getId());
        ctx.startActivity(intent);
	}

	private class ViewHolder {
        public ImageView icon;
        public TextView detail;
        public TextView subDetail;
    }
}
