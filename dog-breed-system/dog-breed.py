from tkinter import *
from windows.main_window import MainWindow

window = Tk()
mywin = MainWindow(window)
window.title('Dog Breed')
window.geometry("960x960+10+10")
window.mainloop()
