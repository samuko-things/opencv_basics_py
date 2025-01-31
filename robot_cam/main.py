import serial
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np

GPIO.setmode(GPIO.BCM)
trigpin = [10,9,11,5,6,13,19,26]
echopin = [24,25,8,7,12,16,20,21]

for j in range(8):
    GPIO.setup(trigpin[j], GPIO.OUT)
    GPIO.setup(echopin[j], GPIO.IN)

ser = serial.Serial('/dev/ttyACM0', 115200)
ser.flushInput()

cap = cv2.VideoCapture(0)
cap.set(3,450)
cap.set(4,200)

def ping(echo, trig):
    GPIO.output(trig, False)
    time.sleep(0.0001)
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)
    pulse_start = time.time()
    
    while GPIO.input(echo) == 0:
        pulse_start = time.time()
        
    while GPIO.input(echo) == 1:
        pulse_end = time.time()
        
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    
    distance = round(distance,2)
    
    return distance


def startblade(x):
    message = 'startblade'
    message = message.encode()
    ser.write(message)
    time.sleep(x)
    
def stopblade(x):
    message = 'stopblade'
    message = message.encode()
    ser.write(message)
    time.sleep(x)
    
def forward(x):
    message = 'f'
    message = message.encode()
    ser.write(message)
    time.sleep(x)
    
def reverse(x):
    message = 'b'
    message = message.encode()
    ser.write(message)
    time.sleep(x)
    
def left(x):
    message = 'l'
    message = message.encode()
    ser.write(message)
    time.sleep(x)
    
def right(x):
    message = 'r'
    message = message.encode()
    ser.write(message)
    time.sleep(x)
    
def stop(x):
    message = 'stop'
    message = message.encode()
    ser.write(message)
    time.sleep(x)


    
try:
    while True:
        ret,frame = cap.read()
    
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
        cv2.imshow('frame', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
        distance1 = ping(echopin[0], trigpin[0])
        #distance2 = ping(echopin[1], trigpin[1])
        distance3 = ping(echopin[2], trigpin[2])
        #distance4 = ping(echopin[3], trigpin[3])
        distance5 = ping(echopin[4], trigpin[4])
        #distance6 = ping(echopin[5], trigpin[5])
        distance7 = ping(echopin[6], trigpin[6])
        #distance8 = ping(echopin[7], trigpin[7])
        print("Sensor1:",distance1, "cm" )
        #print("Sensor2:",distance2, "cm" )
        print("Sensor3:",distance3, "cm" )
        #print("Sensor4:",distance4, "cm" )
        print("Sensor5:",distance5, "cm" )
        #print("Sensor6:",distance6, "cm" )
        print("Sensor7:",distance7, "cm" )
        #print("Sensor8:",distance8, "cm" )

        if distance1 <= 150:
            print("Print Obstacle in front")
            stop(2)
            reverse(2)
            stop(2)
            if distance3 > 100:
                left(1.5)
            elif distance7 > 100:
                right(1.5)
        else:
            forward(2)
            
    cap.release()
    cv2.destroyAllWindows()  
except:
    print("Stop movement")
    stop(5)

