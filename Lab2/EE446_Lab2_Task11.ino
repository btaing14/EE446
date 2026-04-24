#include <Arduino_HS300x.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>

// Thresholds
const float HUMIDITY_JUMP_THRESHOLD = 8.0;
const float TEMPERATURE_RISE_THRESHOLD = 0.3;
const float MAG_SHIFT_THRESHOLD = 25.0;
const int CLEAR_CHANGE_THRESHOLD = 20;

// Cooldown to prevent repeated rapid retriggering
const unsigned long COOLDOWN_MS = 2500;

// Sensor readings
float temperature = 0.0;
float humidity = 0.0;

float mx = 0.0, my = 0.0, mz = 0.0;
float magMetric = 0.0;

int r = 0, g = 0, b = 0, clear = 0;

// Baselines
float baselineTemp = 0.0;
float baselineHumidity = 0.0;
float baselineMag = 0.0;
int baselineClear = 0;

// Event flags
bool humidJump = false;
bool tempRise = false;
bool magShift = false;
bool lightOrColorChange = false;

// Cooldown
unsigned long lastEventTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1500);

  if (!HS300x.begin()) {
    Serial.println("Failed to initialize HS300x sensor.");
    while (1);
  }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }

  delay(1000);

  // Capture baseline using averaged readings
  const int N = 10;
  float tempSum = 0.0;
  float humidSum = 0.0;
  float magSum = 0.0;
  long clearSum = 0;

  for (int i = 0; i < N; i++) {
    tempSum += HS300x.readTemperature();
    humidSum += HS300x.readHumidity();

    if (IMU.magneticFieldAvailable()) {
      IMU.readMagneticField(mx, my, mz);
      magSum += abs(mx) + abs(my) + abs(mz);
    }

    if (APDS.colorAvailable()) {
      APDS.readColor(r, g, b, clear);
      clearSum += clear;
    }

    delay(100);
  }

  baselineTemp = tempSum / N;
  baselineHumidity = humidSum / N;
  baselineMag = magSum / N;
  baselineClear = clearSum / N;

  Serial.println("Task 11 event detector started");
  Serial.println("Baseline captured");
}

void loop() {
  // Read HS300x
  temperature = HS300x.readTemperature();
  humidity = HS300x.readHumidity();

  // Read magnetometer
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
    magMetric = abs(mx) + abs(my) + abs(mz);
  }

  // Read APDS color/light
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clear);
  }

  // Binary event indicators
  humidJump = (humidity - baselineHumidity) > HUMIDITY_JUMP_THRESHOLD;
  tempRise = (temperature - baselineTemp) > TEMPERATURE_RISE_THRESHOLD;
  magShift = abs(magMetric - baselineMag) > MAG_SHIFT_THRESHOLD;
  lightOrColorChange = (abs(clear - baselineClear) > CLEAR_CHANGE_THRESHOLD) || (clear < CLEAR_CHANGE_THRESHOLD);

  // Rule-based event detection
  String finalLabel = "BASELINE_NORMAL";
  static String lastLabel = "BASELINE_NORMAL";

  unsigned long now = millis();
  bool inCooldown = (now - lastEventTime) < COOLDOWN_MS;

  if (!inCooldown) {
    if (humidJump || tempRise) {
      finalLabel = "BREATH_OR_WARM_AIR_EVENT";
      lastEventTime = now;
    }
    else if (magShift) {
      finalLabel = "MAGNETIC_DISTURBANCE_EVENT";
      lastEventTime = now;
    }
    else if (lightOrColorChange) {
      finalLabel = "LIGHT_OR_COLOR_CHANGE_EVENT";
      lastEventTime = now;
    }
    else {
      finalLabel = "BASELINE_NORMAL";
    }

    lastLabel = finalLabel;
  } else {
    finalLabel = lastLabel;
  }

  // Serial output
  Serial.print("raw, rh = ");
  Serial.print(humidity, 2);
  Serial.print(", temp = ");
  Serial.print(temperature, 2);
  Serial.print(", mag = ");
  Serial.print(magMetric, 2);
  Serial.print(", r = ");
  Serial.print(r);
  Serial.print(", g = ");
  Serial.print(g);
  Serial.print(", b = ");
  Serial.print(b);
  Serial.print(", clear = ");
  Serial.println(clear);

  Serial.print("flags, humid_jump = ");
  Serial.print(humidJump ? 1 : 0);
  Serial.print(", temp_rise = ");
  Serial.print(tempRise ? 1 : 0);
  Serial.print(", mag_shift = ");
  Serial.print(magShift ? 1 : 0);
  Serial.print(", light_or_color_change = ");
  Serial.println(lightOrColorChange ? 1 : 0);

  Serial.print("event, ");
  Serial.println(finalLabel);
  Serial.println();

  delay(400);
}
