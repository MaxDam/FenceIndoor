package com.maxdam.fenceindoor.common;

public class CommonStuff {

    public static final String DEFAULTSERVER_PATH = "http://localhost:8080";
    public static final String DEBUG_SERVER_PATH = "http://192.168.1.3:8090";

    //tempo per riprovare la connessione con il server (in secondi)
    public static long RETRY_SLEEP_TIME = 5;

    //timeout per la connessione con il server (millisecondi)
    public static int SERVER_CONNECT_TIMEOUT = 60000;
    public static int SERVER_READ_TIMEOUT = 60000;

    //crash report
    public static final String CRASH_REPORT_URL = DEBUG_SERVER_PATH + "/acra/errorReport.php";
    public static final String CRASH_REPORT_LOGIN ="";
    public static final String CRASH_REPORT_PASSWORD ="";
}
