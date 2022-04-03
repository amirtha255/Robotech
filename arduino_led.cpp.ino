int x;

int LED1 = 13;
int LED2 = 12;
int LED3 = 11;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  
   pinMode(LED1, OUTPUT);
   pinMode(LED2, OUTPUT);
   pinMode(LED3, OUTPUT);
}

void loop() {
  while (!Serial.available());
  x = Serial.readString().toInt();
  //Serial.print(x + 1);

  //# 13, 12, 11 output pins
  if(x==1){
    digitalWrite(LED1, HIGH);    
    digitalWrite(LED2, LOW);    
    digitalWrite(LED3, LOW);    
    //delay(200);        
  }
  else if(x==2) {
    digitalWrite(LED2, HIGH);    
    digitalWrite(LED1, LOW);    
    digitalWrite(LED3, LOW); 
    
  }

  else if(x==3) {
    digitalWrite(LED3, HIGH);    
    digitalWrite(LED2, LOW);    
    digitalWrite(LED1, LOW); 
  }
  else {
    digitalWrite(LED1, LOW);    
    digitalWrite(LED2, LOW);    
    digitalWrite(LED3, LOW); 
  }
  
}
