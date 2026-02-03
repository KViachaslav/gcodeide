import serial
import time

class LaserController:
    def init(self, port, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None

    def connect(self):
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            # Пробуждаем контроллер (GRBL часто перезагружается при подключении)
            self.connection.write(b"\r\n\r\n")
            time.sleep(2)
            self.connection.flushInput()
            print(f"✅ Подключено к {self.port}")
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")

    def send_command(self, gcode):
        if not self.connection or not self.connection.is_open:
            print("Соединение не установлено!")
            return None

        # Очищаем команду и добавляем символ переноса строки
        full_command = gcode.strip() + '\n'
        print(f"📤 Отправка: {full_command.strip()}")
        
        self.connection.write(full_command.encode('utf-8'))
        
        # Ждем ответа от станка (стандарт GRBL возвращает 'ok')
        while True:
            response = self.connection.readline().decode('utf-8').strip()
            if response:
                print(f"📥 Ответ: {response}")
                if response == 'ok':
                    return True
                elif 'error' in response.lower():
                    print(f"⚠️ Ошибка станка: {response}")
                    return False
            
    def close(self):
        if self.connection:
            self.connection.close()
            print("🔌 Соединение закрыто")


laser = LaserController(port='COM3')
laser.connect()

commands = [
    "$X",          # Разблокировать (Unlock) если нужно
    "G21",         # Установка метрической системы (мм)
    "G90",         # Абсолютные координаты
    "G0 X10 Y10",  # Быстрое перемещение
    "M3 S500",     # Включить лазер (мощность 500)
    "G1 X30 F1000",# Рез/движение по линии со скоростью 1000
    "M5",          # Выключить лазер
    "G0 X0 Y0"     # Домой
]

for cmd in commands:
    success = laser.send_command(cmd)
    if not success:
        print("Прерывание цикла из-за ошибки.")
        break

laser.close()