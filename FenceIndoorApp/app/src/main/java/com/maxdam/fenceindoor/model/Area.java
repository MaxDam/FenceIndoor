package com.maxdam.fenceindoor.model;

public class Area {

    private String id;
    private String area;
    private Integer lastScanId;
    private transient Integer color;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getArea() {
        return area;
    }

    public void setArea(String area) {
        this.area = area;
    }

    public Integer getLastScanId() {
        return lastScanId;
    }

    public void setLastScanId(Integer lastScanId) {
        this.lastScanId = lastScanId;
    }

    public Integer getColor() {
        return color;
    }

    public void setColor(Integer color) {
        this.color = color;
    }
}
