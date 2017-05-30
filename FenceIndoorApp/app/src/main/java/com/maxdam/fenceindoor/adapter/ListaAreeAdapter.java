package com.maxdam.fenceindoor.adapter;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.view.DragEvent;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.Gson;
import com.maxdam.fenceindoor.R;
import com.maxdam.fenceindoor.ScanActivity;
import com.maxdam.fenceindoor.common.SessionData;
import com.maxdam.fenceindoor.model.Area;
import com.maxdam.fenceindoor.service.ServiceAddArea;
import com.maxdam.fenceindoor.service.ServiceDeleteArea;
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
         viewHolder.detail.setText(area.getArea());
         viewHolder.subDetail.setText(area.getLastScanId().toString());
         
         //onclick sull'item
         /*convertView.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				onClickListItem(area);
			}
        });*/

        //gestisce l'ontouch
        SwipeListener swipeListener = new SwipeListener();
        swipeListener.setArea(area);
        convertView.setOnTouchListener(swipeListener);
         
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

    //swipe sull'elemento per la cancellazione
    private void onSwipeDeleteItem(final Area area) {
        //ottiene il context
        final Context ctx = ListaAreeAdapter.this.getContext();

        new AlertDialog.Builder(ctx).setMessage("Cancellare " + area.getArea() + "?")
            .setTitle("cancellazione area")
            .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {

                    //comunica con il server per la cancellazione dell'area
                    Intent serviceDeleteAreaIntent = new Intent(ctx, ServiceDeleteArea.class);
                    serviceDeleteAreaIntent.putExtra("area", area.getId());
                    ctx.startService(serviceDeleteAreaIntent);
                }
            })
            .setNegativeButton(android.R.string.no, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                }
            })
            .create().show();
    }

	private class ViewHolder {
        public ImageView icon;
        public TextView detail;
        public TextView subDetail;
    }

    //gestione del touch sull'item
    public class SwipeListener implements View.OnTouchListener {
        private int min_distance = 100;
        private float downX, downY, upX, upY;
        private View v;
        private Area area = null;

        public void setArea(Area area) {
            this.area = area;
        }

        @Override
        public boolean onTouch(View v, MotionEvent event) {
            this.v = v;
            switch(event.getAction()) {
                case MotionEvent.ACTION_DOWN: {
                    downX = event.getX();
                    downY = event.getY();
                    this.onActionDown();
                    return true;
                }
                case MotionEvent.ACTION_UP: {
                    upX = event.getX();
                    upY = event.getY();

                    float deltaX = downX - upX;
                    float deltaY = downY - upY;

                    //HORIZONTAL SCROLL
                    if (Math.abs(deltaX) > Math.abs(deltaY)) {
                        if (Math.abs(deltaX) > min_distance) {
                            // left or right
                            if (deltaX < 0) {
                                this.onLeftToRightSwipe();
                                return true;
                            }
                            if (deltaX > 0) {
                                this.onRightToLeftSwipe();
                                return true;
                            }
                        } else {
                            //not long enough swipe...
                            this.onActionUp();
                            return false;
                        }
                    }
                    //VERTICAL SCROLL
                    else {
                        if (Math.abs(deltaY) > min_distance) {
                            // top or down
                            if (deltaY < 0) {
                                this.onTopToBottomSwipe();
                                return true;
                            }
                            if (deltaY > 0) {
                                this.onBottomToTopSwipe();
                                return true;
                            }
                        } else {
                            //not long enough swipe...
                            this.onActionUp();
                            return false;
                        }
                    }
                    this.onActionUp();
                    return false;
                }
            }
            return true;
        }

        public void onActionUp() {
            if(area != null) {
                onClickListItem(area);
            }
        }
        public void onActionDown() {
            if(area != null) {
                //onClickListItem(area);
            }
        }

        public void onLeftToRightSwipe(){
            if(area != null) {
                onSwipeDeleteItem(area);
            }
        }

        public void onRightToLeftSwipe() {
            if(area != null) {
                onSwipeDeleteItem(area);
            }
        }

        public void onTopToBottomSwipe() {
        }

        public void onBottomToTopSwipe() {
        }
    }
}
