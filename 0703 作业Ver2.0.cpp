int pin = 13;
void setup()
{
pinMode(pin, OUTPUT);
}
void loop()
{
  char str[10000];
  int i,t,n=0; 
  while (Serial.available() > 0) 
    {
      str[n]=Serial.read();
      delay(2);
      n++;
    }
  ;
   for(i=0;i<n;i++)
   {
    switch(str[i])
      {
        case 'a':{dot();dash();break;}//a
        case 98:{dash();dot();dot();dot();break;}//b
        case 99:{dash();dot();dash();dot();break;}//c
        case 100:{dash();dot();dot();break;}//d
        case 101:{dot();break;}//e
        case 102:{dot();dot();dash();dot();break;}//f
        case 103:{dash();dash();dot();break;}//g
        case 104:{dot();dot();dot();dot();break;}//h
        case 105:{dot();dot();break;}//i
        case 106:{dot();dash();dash();dash();break;}//j
        case 107:{dash();dot();dash();break;}//k
        case 108:{dot();dash();dot();dot();break;}//l
        case 109:{dash();dash();break;}//m
        case 110:{dash();dot();break;}//n
        case 111:{dash();dash();dash();break;}//o
        case 112:{dot();dash();dash();dot();break;}//p
        case 113:{dash();dash();dot();dash();break;}//q
        case 114:{dot();dash();dot();break;}//r
        case 115:{dot();dot();dot();break;}//s
        case 116:{dash();break;}//t
        case 117:{dot();dot();dash();break;}//u
        case 118:{dot();dot();dot();dash();break;}
        case 119:{dot();dash();dash();break;}//w
        case 120:{dash();dot();dot();dash();break;}//x
        case 121:{dash();dot();dash();dash();break;}//y
        case 122:{dash();dash();dot();dot();break;}//z
  
        }
    }
delay(3000);
}



void dot()
{
digitalWrite(pin, HIGH);
delay(250);
digitalWrite(pin, LOW);
delay(250);
}



void dash()
{
digitalWrite(pin, HIGH);
delay(1000);
digitalWrite(pin, LOW);
delay(250);
}
