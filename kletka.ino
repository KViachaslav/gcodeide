#include <Arduino.h>
#include <U8g2lib.h>
#include <math.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#define ONE_WIRE_BUS 14
OneWire oneWire(ONE_WIRE_BUS);

DallasTemperature sensors(&oneWire);

int total_devices;
DeviceAddress sensor_address; 
#define TEMPER_FONT_HEIGHT 10 
#define MAX_SPEED 60
U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, 12, 14, U8X8_PIN_NONE);

void drawPie(int x0, int y0, int r, int start_deg, int end_deg) {
  for (int i = start_deg; i <= end_deg; i++) {
   
    float rad = i * 0.0174532925; 
    int x = x0 + r * cos(rad);
    int y = y0 + r * sin(rad);
    u8g2.drawLine(x0, y0, x, y); 
  }
}

void drawMainWindow(float speed, int probeg, float t1, float t2, float t3) {
     u8g2.clearBuffer();

    char tempString[10];
    u8g2.setFont(u8g2_font_ncenB10_tr); 

    snprintf(tempString, sizeof(tempString), "%.2f °C", t1);
    u8g2.drawStr(99, 16+TEMPER_FONT_HEIGHT, tempString);
    snprintf(tempString, sizeof(tempString), "%.2f °C", t2);
    u8g2.drawStr(99, 33+TEMPER_FONT_HEIGHT, tempString);
    snprintf(tempString, sizeof(tempString), "%.2f °C", t3);
    u8g2.drawStr(99, 50+TEMPER_FONT_HEIGHT, tempString);

    u8g2.setFont(u8g2_font_ncenB14_tr); 
    snprintf(tempString, sizeof(tempString), "%d km.", probeg);
    u8g2.drawStr(68, 14, tempString);
    snprintf(tempString, sizeof(tempString), "%.2f m/min", speed);
    u8g2.drawStr(4, 14, tempString);

    int end_deg = speed/ MAX_SPEED * 180;
    drawPie(48, 64, 48, 0, end_deg);
    u8g2.sendBuffer();
}

void setup() {
  u8g2.begin();
  Serial.begin(115200);
  sensors.begin();
  
  total_devices = sensors.getDeviceCount();
  
  Serial.print("Locating devices...");
  Serial.print("Found ");
  Serial.print(total_devices, DEC);
  Serial.println(" devices.");

  for(int i=0;i<total_devices; i++){
    if(sensors.getAddress(sensor_address, i)){
      Serial.print("Found device ");
      Serial.print(i, DEC);
      Serial.print(" with address: ");
      printAddress(sensor_address);
      Serial.println();
    } else {
      Serial.print("Found device at ");
      Serial.print(i, DEC);
      Serial.print(" but could not detect address. Check circuit connection!");
    }
  }
}

void loop() {
    sensors.requestTemperatures(); 
  
  for(int i=0;i<total_devices; i++){
    
    if(sensors.getAddress(sensor_address, i)){
      
      Serial.print("Temperature for device: ");
      Serial.println(i,DEC);
     
      float temperature_degreeCelsius = sensors.getTempC(sensor_address);
      Serial.print("Temp (degree celsius): ");
      Serial.println(temperature_degreeCelsius);
  
    }
  }
  delay(10000);
    drawMainWindow(12.2,1230,21.3,21.0,20.8)
    delay(2000);
}

void printAddress(DeviceAddress deviceAddress) {
  for (uint8_t i = 0; i < 8; i++){
    if (deviceAddress[i] < 16) Serial.print("0");
      Serial.print(deviceAddress[i], HEX);
  }
}