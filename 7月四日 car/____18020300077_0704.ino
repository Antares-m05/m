void setup(){
   pinMode(5,OUTPUT) ;
   pinMode(6,OUTPUT) ;
   pinMode(9,OUTPUT) ;
   pinMode(10,OUTPUT) ;
  Serial.begin(9600);
  
}

int income = 0;
void loop()
{
  {
    if (Serial.available()>0){
      income = Serial.read();
      switch(income){
        case'f':
          forward();
          break;
        case'b':
          backward();
          break;
        case'l':
          left();
        for(int i = 0;i<=5;i++){ 
            digitalWrite(12,HIGH);
          delay(500);
            digitalWrite(12,LOW);
            delay(500);
            if (Serial.available()>0){break;}
            }
            
          
        case'r':
          right();
          for(int i = 0;i<=5;i++){ 
            digitalWrite(2,HIGH);
          delay(500);
            digitalWrite(2,LOW);
            delay(500);
            if (Serial.available()>0){break;}
            }
          break;
        default:
          break;
      }
      
    }
  }
}


void forward()
{
digitalWrite(5,HIGH);
digitalWrite(6,LOW);
digitalWrite(10,LOW);
digitalWrite(9,HIGH);
}


void backward()
{
digitalWrite(6,HIGH);
digitalWrite(5,LOW);
digitalWrite(9,LOW);
digitalWrite(10,HIGH);
}


void right()
{
digitalWrite(6,LOW);
digitalWrite(5,HIGH);
digitalWrite(9,LOW);
digitalWrite(10,HIGH);
}





void left()
{
digitalWrite(6,HIGH);
digitalWrite(5,LOW);
digitalWrite(9,HIGH);
digitalWrite(10,LOW);
}

void lflash()
{
  digitalWrite(12,HIGH);
  delay(1000);
 
}
