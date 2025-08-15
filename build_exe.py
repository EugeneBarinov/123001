#!/usr/bin/env python3
"""
Сборка exe файла для Crypto AI Analytics
Использует PyInstaller для создания исполняемого файла
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_pyinstaller():
    """Проверяет установлен ли PyInstaller"""
    try:
        import PyInstaller
        print("PyInstaller найден")
        return True
    except ImportError:
        print("PyInstaller не найден")
        return False

def install_pyinstaller():
    """Устанавливает PyInstaller"""
    print("Установка PyInstaller...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller установлен")
        return True
    except subprocess.CalledProcessError:
        print("Ошибка установки PyInstaller")
        return False

def build_exe():
    """Собирает exe файл"""
    print("Начинаю сборку exe файла...")
    
    # Очищаем предыдущие сборки
    dist_dir = Path("dist")
    build_dir = Path("build")
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print("Очищен каталог dist")
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("Очищен каталог build")
    
    # Простая команда сборки
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name=CryptoAI",
        "main.py"
    ]
    
    try:
        print("Запуск PyInstaller...")
        print("Это может занять несколько минут...")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Exe файл успешно создан!")
            
            # Проверяем результат
            exe_path = dist_dir / "CryptoAI.exe"
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"Файл: {exe_path}")
                print(f"Размер: {size_mb:.1f} MB")
                print(f"Расположение: {exe_path.absolute()}")
                
                return True
            else:
                print("Exe файл не найден после сборки")
                return False
        else:
            print("Ошибка сборки")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения PyInstaller: {e}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False

def main():
    """Главная функция"""
    print("Сборка exe файла для Crypto AI Analytics")
    print("=" * 60)
    
    # Проверяем PyInstaller
    if not check_pyinstaller():
        print("Устанавливаю PyInstaller...")
        if not install_pyinstaller():
            print("Не удалось установить PyInstaller")
            return False
    
    # Собираем exe
    if build_exe():
        print("\nСборка завершена успешно!")
        print("Теперь вы можете запустить CryptoAI.exe")
        return True
    else:
        print("\nОшибка сборки")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nГотово! Приложение можно запускать")
    else:
        print("\nОшибка сборки")
    
    input("\nНажмите Enter для выхода...")
