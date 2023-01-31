# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:54:36 2022

@author: KM
"""

import sys
from PyQt5 import QtWidgets, uic

from ilkTasarim import Ui_MainWindow


def dortIslemYap(secim, sayi1, sayi2):
    if secim == '+':
        sonuc = sayi1 + sayi2
    elif secim == '-':
        sonuc = sayi1 - sayi2
    elif secim == '*':
        sonuc = sayi1 * sayi2
    elif secim == '/':
        sonuc = sayi1 / sayi2
    elif secim == '%':
        sonuc = sayi1 % sayi2

    return sonuc


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.lblSonuc.hide()
        self.pushButton.clicked.connect(self.islemYap)

    def islemYap(self):

        if self.txtSayi1.text().isnumeric() and self.txtSayi2.text().isnumeric():
            sayi1 = int(self.txtSayi1.text())
            sayi2 = int(self.txtSayi2.text())
            secim = self.cmbSecim.currentText()
            sonuc = dortIslemYap(secim, sayi1, sayi2)
            self.lblSonuc.setText("İslem sonucu:" + str(sonuc))
            self.lblSonuc.setStyleSheet("color: black")

        else:
            self.lblSonuc.setText("Lütfen sayı giriniz")
            self.lblSonuc.setStyleSheet("color: #FF0000")

        self.lblSonuc.show()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()