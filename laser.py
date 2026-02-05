import serial
import time
class LaserController:
    def __init__(self, port, baudrate=115200, timeout=1):
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
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False

    def send_command(self, gcode):
        if not self.connection or not self.connection.is_open:
            print("Соединение не установлено!")
            return False

        # Очищаем команду и добавляем символ переноса строки
        full_command = gcode.strip() + '\n'
        
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