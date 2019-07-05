class Morse
{
  public:
    Morse(int pin);
    void dot();
    void dash();
    void c_space();
    void w_space();
  private:
    int _pin;
    int _dottime;
};


Morse::Morse(int pin)
{
  pinMode(pin,OUTPUT);
  _pin=pin;
  _dottime=100;
}

void Morse::dot()
{
  digitalWrite(_pin,HIGH);
  delay(_dottime);
  digitalWrite(_pin,LOW);
  delay(_dottime);
}

void Morse::dash()
{
  digitalWrite(_pin,HIGH);
  delay(_dottime*4);
  digitalWrite(_pin,LOW);
  delay(_dottime);
}

void Morse::c_space()
{
  digitalWrite(_pin,LOW);
  delay(_dottime*3);
}

void Morse::w_space()
{
  digitalWrite(_pin,LOW);
  delay(_dottime*7);
}

Morse morse(13);
char MORSE[][5] = {
  {'.', '-', '*', '*'}, 
  {'-', '.', '.', '.'}, 
  {'-', '.', '-', '.'}, 
  {'-', '.', '.', '*'},
  {'.', '*', '*', '*'}, 
  {'.', '.', '-', '.'}, 
  {'-', '-', '.', '*'}, 
  {'.', '.', '.', '.'}, 
  {'.', '.', '*', '*'}, 
  {'.', '-', '-', '-'}, 
  {'-', '.', '-', '*'},
  {'.', '-', '.', '.'}, 
  {'-', '-', '*', '*'},
  {'-', '.', '*', '*'}, 
  {'-', '-', '-', '*'}, 
  {'.', '-', '-', '.'}, 
  {'-', '-', '.', '-'},
  {'.', '-', '.', '*'},
  {'.', '.', '.', '*'},
  {'-', '*', '*', '*'},
  {'.', '.', '-', '*'},
  {'.', '.', '.', '-'},
  {'.', '-', '-', '*'},
  {'-', '.', '.', '-'},
  {'-', '.', '-', '-'},
  {'-', '-', '.', '.'},
};

void setup()
{
  Serial.begin(9600);
}

void loop()
{
  String ls = "";
  String morse_s = "";
  int i, t , temp = 0;
  int n = 0;
  while(Serial.available()>0)
  {
    temp = 1;
    ls += char(Serial.read());
    delay(1); 
    n++;
  }
   if(temp)
  {
    for(i=0;i<n;i++)
    {
      for(t=0;t<4;t++)
      {
        if((ls[i]>=97)&&(ls[i] <= 122))
        {
          morse_s += char(MORSE[int(ls[i] - 97)][t]);
        }
        if(ls[i]==13)
        {
          morse_s += char(MORSE[27][t]);
        }
        if(ls[i]==32)
        {
          morse_s += char(MORSE[28][t]);
        }
      }
      morse_s += ' ';
    }
    Serial.println(morse_s);
    for(i = 0; morse_s[i]!='\0' ; i++)
    {
      if(morse_s[i] == '.')morse.dot();
      else if(morse_s[i] == '-')morse.dash();
      else if(morse_s[i] == ' ')morse.w_space();
      if (morse_s[i] != ' ' && ls[i] != '*')morse.c_space();
    }
    delay(1);
  }
