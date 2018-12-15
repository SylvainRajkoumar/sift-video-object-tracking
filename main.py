from PyQt5.QtWidgets import QApplication
from view import View
from controller import Controller
import sys

def main():
    app = QApplication(sys.argv)
    a = View()
    b = Controller(a)
    a.setControl(b)
    a.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()