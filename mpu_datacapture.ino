#include <Adafruit_Sensor.h>
#include <Adafruit_MPU6050.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(9600);
  
  // 尝试初始化，明确指定设备ID
  if (!mpu.begin(0x68, &Wire, 0, 400000, 0x70)) { // 最后参数是期望的WHO_AM_I值
    Serial.println("Failed to initialize MPU!");
    while(1) delay(10);
  }
  
  // 配置参数
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  Serial.println("MPU6500 initialized!");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  Serial.print("Accel X:");
  Serial.print(a.acceleration.x);
  Serial.print(", Y:");
  Serial.print(a.acceleration.y);
  Serial.print(", Z:");
  Serial.print(a.acceleration.z);
  Serial.println(" m/s^2");
  
  Serial.print("Gyro X:");
  Serial.print(g.gyro.x);
  Serial.print(", Y:");
  Serial.print(g.gyro.y);
  Serial.print(", Z:");
  Serial.print(g.gyro.z);
  Serial.println(" rad/s");
  
  Serial.print("Temperature:");
  Serial.print(temp.temperature);
  Serial.println(" degC");
  
  delay(500);
}