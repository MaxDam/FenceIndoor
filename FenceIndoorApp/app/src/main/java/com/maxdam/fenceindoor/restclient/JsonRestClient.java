package com.maxdam.fenceindoor.restclient;

import android.util.Base64;

import com.google.gson.Gson;
import com.maxdam.fenceindoor.common.CommonStuff;
import com.maxdam.fenceindoor.util.StringUtil;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.security.SecureRandom;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.TimeZone;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.KeyManager;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;

public class JsonRestClient {

	private StringBuffer url = new StringBuffer();
	private List<String> queryParams = new ArrayList<String>();
	private String outputString =null;
	private Object request;
	private String requestBodyString;
	private Gson mapper;
	private String basicAuthStringEnc = null;
	
	public static JsonRestClient newInstance(String host){
		JsonRestClient instance = new JsonRestClient();
		instance.url = new StringBuffer(host);
		instance.request = null;
		return instance;
	}

	private JsonRestClient() {
		mapper = new Gson();
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");
		dateFormat.setTimeZone(TimeZone.getTimeZone("GMT"));
	}

	public JsonRestClient addPath(String path){
		url.append("/").append(path);
		return this;
	}

	public JsonRestClient addQueryParam(String paramKey, String paramValue) throws UnsupportedEncodingException {
		paramKey = URLEncoder.encode(paramKey, "UTF-8");
		paramValue = URLEncoder.encode(paramValue, "UTF-8");
		queryParams.add(String.format("%s=%s", paramKey, paramValue));
		return this;
	}

	public JsonRestClient addQueryParam(String paramKey, int paramValue) throws UnsupportedEncodingException {
		paramKey = URLEncoder.encode(paramKey, "UTF-8");
		queryParams.add(String.format("%s=%s", paramKey, paramValue));
		return this;
	}
	
	public JsonRestClient addQueryParam(String paramKey, long paramValue) throws UnsupportedEncodingException {
		paramKey = URLEncoder.encode(paramKey, "UTF-8");
		queryParams.add(String.format("%s=%s", paramKey, paramValue));
		return this;
	}

	public JsonRestClient addQueryParam(String paramKey, float paramValue) throws UnsupportedEncodingException {
		paramKey = URLEncoder.encode(paramKey, "UTF-8");
		queryParams.add(String.format("%s=%s", paramKey, paramValue));
		return this;
	}

	public JsonRestClient addQueryParam(String paramKey, double paramValue) throws UnsupportedEncodingException {
		paramKey = URLEncoder.encode(paramKey, "UTF-8");
		queryParams.add(String.format("%s=%s", paramKey, paramValue));
		return this;
	}
	
	public JsonRestClient addParam(String param){
		url.append("/").append(param);
		return this;
	}

	public JsonRestClient addParam(int param){
		url.append("/").append(param);
		return this;
	}

	public JsonRestClient addParam(long param){
		url.append("/").append(param);
		return this;
	}

	public JsonRestClient addParam(boolean param){
		url.append("/").append(param);
		return this;
	}
	
	public JsonRestClient addParam(float param){
		url.append("/").append(param);
		return this;
	}

	public JsonRestClient addParam(double param){
		url.append("/").append(param);
		return this;
	}

	//ritorna l'url completo
	private String getURL() {
		String returnedUrl = this.url.toString();

		//se ci sono parametri in get li aggiunge
		if(queryParams.size() == 0) {
			returnedUrl = this.url.toString();
		}
		else {
			returnedUrl += String.format("?%s", StringUtil.join(queryParams, "&"));
		}
		return returnedUrl;
	}
	
	//set la stringa di autenticazione
	public JsonRestClient setBasicAuth(String name, String password) {
		String authString = name + ":" + password;
		System.out.println("auth string: " + authString);
		byte[] authEncBytes = Base64.encode(authString.getBytes(), Base64.DEFAULT);
		this.basicAuthStringEnc = new String(authEncBytes);
		return this;
	}
	
	public JsonRestClient setRequestBody(Object request){
		this.request = request;
		return this;
	}

	public JsonRestClient get() throws Exception {

		if (this.url.toString().toLowerCase().startsWith("https")){
			TrustManager[] tm = new TrustManager[] { new NativeTrustManager() };
			SSLContext sslcontext = SSLContext.getInstance ("SSL");
			sslcontext.init( new KeyManager[0], tm, new SecureRandom( ) );
			SSLSocketFactory sslSocketFactory;
			sslSocketFactory = (SSLSocketFactory) sslcontext.getSocketFactory ();

			HttpsURLConnection.setDefaultSSLSocketFactory(sslSocketFactory);
			HttpsURLConnection.setDefaultHostnameVerifier(new HostnameVerifier() {
				public boolean verify(String hostname, SSLSession session) {
					return true;  
				}  
			});  
		}
		
		URL url = new URL(getURL());
		HttpURLConnection conn = (HttpURLConnection) url.openConnection();
		conn.setConnectTimeout(CommonStuff.SERVER_CONNECT_TIMEOUT);
		conn.setReadTimeout(CommonStuff.SERVER_READ_TIMEOUT);
		conn.setRequestMethod("GET");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "application/json");
        if(this.basicAuthStringEnc != null) {
        	conn.setRequestProperty("Authorization", "Basic " + this.basicAuthStringEnc);
        }

		if (conn.getResponseCode() >= 300) {
			String errorMsg = String.format("ERRORE INVOCAZIONE METODO %s - HTTPS ERROR CODE %s CON MESSAGGIO %s", this.toString(), conn.getResponseCode(), getOutputString(conn.getErrorStream()));
			throw new Exception(conn.getResponseCode()+"", new Exception(errorMsg));
		}

		setOutputString(getOutputString(conn.getInputStream()));
		conn.disconnect();
		
		return this;
	}

	public JsonRestClient post() throws Exception {

		if (this.url.toString().toLowerCase().startsWith("https")){
			TrustManager[] tm = new TrustManager[] { new NativeTrustManager() };
			SSLContext sslcontext = SSLContext.getInstance ("SSL");
			sslcontext.init( new KeyManager[0], tm, new SecureRandom( ) );
			SSLSocketFactory sslSocketFactory;
			sslSocketFactory = (SSLSocketFactory) sslcontext.getSocketFactory ();

			HttpsURLConnection.setDefaultSSLSocketFactory(sslSocketFactory);
			HttpsURLConnection.setDefaultHostnameVerifier(new HostnameVerifier() {
				public boolean verify(String hostname, SSLSession session) {
					return true;  
				}  
			});  
		}
		
		URL url = new URL(getURL());
		HttpURLConnection conn = (HttpURLConnection)url.openConnection();
		conn.setConnectTimeout(CommonStuff.SERVER_CONNECT_TIMEOUT);
		conn.setReadTimeout(CommonStuff.SERVER_READ_TIMEOUT);
		conn.setRequestMethod("POST");
		conn.setRequestProperty("Content-Type", "application/json");
		conn.setRequestProperty("Accept", "application/json");
        if(this.basicAuthStringEnc != null) {
        	conn.setRequestProperty("Authorization", "Basic " + this.basicAuthStringEnc);
        }

		if (request != null){
	            OutputStream os = conn.getOutputStream();
	            
	            if(request instanceof String) {
	                    //inserisce direttamente la stringa
	                    requestBodyString = request.toString();
	            }
	            else {
	                    //marshalling json object
	                    requestBodyString = mapper.toJson(request);
	            }
	            
	            os.write(requestBodyString.getBytes());
	            os.flush();
	    }

		if (conn.getResponseCode() >= 300) {
			String errorMsg = String.format("ERRORE INVOCAZIONE METODO %s - HTTPS ERROR CODE %s CON MESSAGGIO %s", this.toString(), conn.getResponseCode(), getOutputString(conn.getErrorStream()));
			throw new Exception(conn.getResponseCode()+"", new Exception(errorMsg));
		}
		
		StringBuffer output = getBodyResponse(conn);
        conn.disconnect();
        
        outputString = output.toString();
        return this;
	}

	//ottiene l'output
    private String getOutputString(InputStream inputStream) throws IOException {
            String line;
            StringBuffer output = new StringBuffer();
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
            while ((line = br.readLine()) != null) {
                    output.append(line + "\n");
            }
            return output.toString();
    }

    private StringBuffer getBodyResponse(HttpURLConnection conn)
                    throws IOException {
            String line;
            StringBuffer output = new StringBuffer();
            BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            while ((line = br.readLine()) != null) {
                    output.append(line);
            }
            return output;
    }

	public String getOutputString() {
	    return outputString;
	}
	
	public void setOutputString(String outputString) {
	    this.outputString = outputString;
	}
	
	public String toString() {
		return getURL();
	}
}
