#include <PDM.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>

// Microphone
short sampleBuffer[256];
volatile int samplesRead = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

// Thresholds
const int SOUND_THRESHOLD = 100;
const float MOTION_THRESHOLD = 20.0;
const int PROX_NEAR_THRESHOLD = 100;
const int CLEAR_DARK_THRESHOLD = 20;

// State
int micLevel = 0;
float gx = 0.0, gy = 0.0, gz = 0.0;
float motionMetric = 0.0;
int proximityValue = 0;
int r = 0, g = 0, b = 0, clear = 0;

bool soundDetected = false;
bool darkDetected = false;
bool movingDetected = false;
bool nearDetected = false;

void setup() {
  Serial.begin(115200);
  delay(1500);

  // Microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM microphone.");
    while (1);
  }

  // IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  // APDS9960: proximity + color/light
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }

  Serial.println("Task 10 workspace classifier started");
}

void loop() {
  // Read microphone
  if (samplesRead > 0) {
    long sum = 0;
    for (int i = 0; i < samplesRead; i++) {
      sum += abs(sampleBuffer[i]);
    }
    micLevel = sum / samplesRead;
    samplesRead = 0;
  }

  // Read gyroscope for motion
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
    motionMetric = abs(gx) + abs(gy) + abs(gz);
  }

  // Read proximity
  if (APDS.proximityAvailable()) {
    proximityValue = APDS.readProximity();
  }

  // Read ambient light
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clear);
  }

  // Binary decisions
  soundDetected = (micLevel > SOUND_THRESHOLD);

  movingDetected = (motionMetric > MOTION_THRESHOLD);

  nearDetected = (proximityValue < PROX_NEAR_THRESHOLD);

  darkDetected = (clear < CLEAR_DARK_THRESHOLD);

  // Rule-based classification
  String finalLabel = "UNCLASSIFIED";

  if (!soundDetected && !darkDetected && !movingDetected && !nearDetected) {
    finalLabel = "QUIET_BRIGHT_STEADY_FAR";
  }
  else if (soundDetected && !darkDetected && !movingDetected && !nearDetected) {
    finalLabel = "NOISY_BRIGHT_STEADY_FAR";
  }
  else if (!soundDetected && darkDetected && !movingDetected && nearDetected) {
    finalLabel = "QUIET_DARK_STEADY_NEAR";
  }
  else if (soundDetected && !darkDetected && movingDetected && nearDetected) {
    finalLabel = "NOISY_BRIGHT_MOVING_NEAR";
  }

  Serial.print("raw, mic = ");
  Serial.print(micLevel);
  Serial.print(", clear = ");
  Serial.print(clear);
  Serial.print(", motion = ");
  Serial.print(motionMetric, 2);
  Serial.print(", prox = ");
  Serial.println(proximityValue);

  Serial.print("flags, sound = ");
  Serial.print(soundDetected ? 1 : 0);
  Serial.print(", dark = ");
  Serial.print(darkDetected ? 1 : 0);
  Serial.print(", moving = ");
  Serial.print(movingDetected ? 1 : 0);
  Serial.print(", near = ");
  Serial.println(nearDetected ? 1 : 0);

  Serial.print("state, ");
  Serial.println(finalLabel);
  Serial.println();

  delay(300);
}
