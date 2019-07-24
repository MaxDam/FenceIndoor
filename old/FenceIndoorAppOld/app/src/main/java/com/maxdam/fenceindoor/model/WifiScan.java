package com.maxdam.fenceindoor.model;

public class WifiScan {

    private String wifiName;
    private Integer wifiLevel;
    private transient Integer color;

    public String getWifiName() {
        return wifiName;
    }

    public void setWifiName(String wifiName) {
        this.wifiName = wifiName;
    }

    public Integer getWifiLevel() {
        return wifiLevel;
    }

    public void setWifiLevel(Integer wifiLevel) {
        this.wifiLevel = wifiLevel;
    }

    public Integer getColor() {
        return color;
    }

    public void setColor(Integer color) {
        this.color = color;
    }

}
