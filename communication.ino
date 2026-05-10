
int correction = 0;

void setup() {
  Serial.begin(9600);
  
  while (!Serial) {
    
  }
}

void loop() {
  if (Serial.available()) {
    String message = Serial.readStringUntil('\n');
    amount = message.toInt();

    if (correction > 0) {
      Serial.println("Move to the right");
    }

    else if (correction < 0) {
      Serial.println("Move to the left");
    }
    
    Serial.println(message);
  }
}
