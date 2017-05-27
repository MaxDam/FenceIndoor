package com.maxdam.fenceindoor.model;

/**
 * Created by max on 25/05/17.
 */
public class WifiScan {

    private String name;
    private Integer level;
    private transient Integer color;
    private String area;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getLevel() {
        return level;
    }

    public void setLevel(Integer level) {
        this.level = level;
    }

    public Integer getColor() {
        return color;
    }

    public void setColor(Integer color) {
        this.color = color;
    }

    public String getArea() {
        return area;
    }

    public void setAreaName(String area) {
        this.area = area;
    }
}
