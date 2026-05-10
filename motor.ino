  // Right motor
const int In1 = 10;
const int In2 = 11; 

// Left motor
const int In3 = 8;
const int In4 = 9; 

const int EnA = 6; 
//const int EnB = 5; 

void setup()
{
  pinMode(In1, OUTPUT);
  pinMode(In2, OUTPUT);
  pinMode(In3, OUTPUT);
  pinMode(In4, OUTPUT);
  pinMode(EnA, OUTPUT); 
}

void goStraight()   //run both motors in the same direction
{
  // Turn on motor (right)
  digitalWrite(In1, HIGH);
  digitalWrite(In2, LOW);

  // Turn on motor (left)
  digitalWrite(In3, HIGH);
  digitalWrite(In4, LOW);
  
  analogWrite(EnA, 50);
  //analogWrite(EnB, 150);
}

void loop()
{
  goStraight();
  delay(1000);
}
