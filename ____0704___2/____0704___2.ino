void setup()
{
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  Serial.begin(9600);
}

void loop()
{
 
  char in;
  if(Serial.available()>0)  in = (Serial.read());
  switch(in){
    case '1' :{ digitalWrite(5,LOW);
  digitalWrite(3,LOW); 
  digitalWrite(4,LOW);
  digitalWrite(2,HIGH);break;}
  
    case'2':{digitalWrite(2,LOW);
  digitalWrite(5,LOW); 
  digitalWrite(4,LOW);
  digitalWrite(3,HIGH);break;}
    
    case '3':{
  digitalWrite(2,HIGH);
  digitalWrite(5,LOW); 
  digitalWrite(4,LOW);
  digitalWrite(3,HIGH);break;
}
    case'4' :{
  digitalWrite(2,LOW);
  digitalWrite(3,LOW); 
  digitalWrite(5,LOW);
  digitalWrite(4,HIGH);break;
}
    
    case'5':{
  digitalWrite(2,HIGH);
  digitalWrite(3,LOW); 
  digitalWrite(5,LOW);
  digitalWrite(4,HIGH);break;
}
    case'6':{
  digitalWrite(2,LOW);
  digitalWrite(3,HIGH); 
  digitalWrite(5,LOW);
  digitalWrite(4,HIGH);break;
}
    case'7':{
  digitalWrite(2,HIGH);
  digitalWrite(3,HIGH); 
  digitalWrite(4,HIGH);
  digitalWrite(5,LOW);break;
}
    case'8':{
  digitalWrite(2,LOW);
  digitalWrite(3,LOW); 
  digitalWrite(4,LOW);
  digitalWrite(5,HIGH);break;
}
    case'9':{
  digitalWrite(2,HIGH);
  digitalWrite(3,LOW); 
  digitalWrite(4,LOW);
  digitalWrite(5,HIGH);break;
} 
    case'0':{
  digitalWrite(2,LOW);
  digitalWrite(3,LOW); 
  digitalWrite(4,LOW);
  digitalWrite(5,LOW);break;
}
  
  }
   
 
   
}  
