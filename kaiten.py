import time
import serial   #https://www.cooptec.jp/archives/1511

# シリアル通信の設定
ser = serial.Serial('COM3', 9600)  # ポート名に応じて変更

ser.flushInput()
ser.flushOutput()
rotate = True
timer = 5   #何秒回転させるか、ここは実物みて距離計算するしかない。直径＊π　加速度も考えないと・・・
timeS=time.time()
NowTime = 0

while NowTime>=timer:
    ser.write(b's30\n')  # 角度設定(これで速さが変わるらしい)
    NowTime = time.time()-timeS
    
    
   
