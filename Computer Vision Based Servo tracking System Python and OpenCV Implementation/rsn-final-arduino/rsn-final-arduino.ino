#include <Servo.h>

#define PAN_SERVO_PIN 9
#define TILT_SERVO_PIN 10
#define CENTER_VAL 1000

int pan_pos = 90;
int tilt_pos = 90;

Servo pan;
Servo tilt;

void setup() {
  Serial.begin(115200);
  pan.attach(PAN_SERVO_PIN);
  tilt.attach(TILT_SERVO_PIN);
}


// map value to be between 0, 180
int adjust_val(int val) {
  return min(150, max(30, val));
}

String readLastString() {
  String s = "";
  while (Serial.available() > 0) {
    s = Serial.readStringUntil('\n');
  }
  return s;
}

void loop() {
  if (Serial.available() > 0) {
    String input = readLastString();
    int separatorIndex = input.indexOf(';');
    if (separatorIndex != -1) {
      int pos1 = input.substring(0, separatorIndex).toInt();
      int pos2 = input.substring(separatorIndex + 1).toInt();

      if (pos1 == CENTER_VAL) {
        pos1 = 90 - pan_pos;
      }
      if (pos2 == CENTER_VAL) {
        pos2 = 90 - tilt_pos;
      }


      pan_pos = adjust_val(pan_pos + pos1);
      tilt_pos = adjust_val(tilt_pos + pos2);

      Serial.println("pan: " + String(pan_pos) + " tilt: " + String(tilt_pos));

      pan.write(pan_pos);
      tilt.write(tilt_pos);

      
    }
  }
}
