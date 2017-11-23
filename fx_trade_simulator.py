#! /usr/bin/env python3
# coding: utf-8
"""
This module is put label for FX-history-data.
This label means 0:don't move 1:buy 2:buy full force 3:sell 4:sell full force
"""
import csv
import math

class Asset:
    """ 資産状況
    """
    def __init__(self, startMoney):
        self.startmoney = startMoney
        self.currentmoney = startMoney
        self.stocks = []
        self.profitandloss = 0
class Stock:
    """ 継続中の取引クラス
    """
    spread = 0.003
    leverage = 10
    def __init__(self, datetime, price, amount, position, wp):
        self.datetime = datetime
        self.price = price
        self.amount = amount
        self.position = position
        self.reportStart(wp)

    def settlement(self, ast, datetime, price, wp):
        if self.position == 'B':
            profitandloss = math.floor((price - self.price - self.spread) * self.amount)
        elif self.position == 'S':
            profitandloss = math.floor((self.price - price - self.spread) * self.amount)
        ast.currentmoney = ast.currentmoney + math.floor(self.price * self.amount / self.leverage) + profitandloss
        ast.stocks.pop(0)
        self.reportEnd(datetime, price, profitandloss, wp)

    def reportStart(self, wp):
        report = []
        report.append(self.datetime)
        report.append('')
        report.append(str(self.price))
        report.append('')
        report.append('')
        wp.writerow(report)

    def reportEnd(self, edatetime, eprice, profitandloss, wp):
        report = []
        report.append(self.datetime)
        report.append(edatetime)
        report.append(str(self.price))
        report.append(str(eprice))
        report.append(str(profitandloss))
        wp.writerow(report)

class Trade:
    """ 新規取引クラス
    """
    leverage = 10
    def buy(self, ast, datetime, price):
        cap = 0
        if ast.currentmoney != 0:
            cap = math.floor(float(ast.currentmoney * self.leverage / price))
        if cap > 0:
            ast.currentmoney = ast.currentmoney - (math.floor(cap / 3 / self.leverage) * price)
            ast.stocks.append(Stock(datetime, price, math.floor(cap / 3), 'B', writer_report))

    def buyAll(self, ast, datetime, price):
        cap = 0
        if ast.currentmoney != 0:
            cap = math.floor(float(ast.currentmoney * self.leverage / price))
        if cap > 0:
            ast.currentmoney = ast.currentmoney - (cap * price / self.leverage)
            ast.stocks.append(Stock(datetime, price, cap, 'B', writer_report))

    def sell(self, ast, datetime, price):
        cap = 0
        if ast.currentmoney != 0:
            cap = math.floor(float(ast.currentmoney * self.leverage / price))
        if cap != 0:
            ast.currentmoney = ast.currentmoney - (math.floor(cap / 3 / self.leverage) * price)
            ast.stocks.append(Stock(datetime, price, math.floor(cap / 3), 'S', writer_report))

    def sellAll(self, ast, datetime, price):
        cap = 0
        if ast.currentmoney != 0:
            cap = math.floor(float(ast.currentmoney * self.leverage / price))
        if cap != 0:
            ast.currentmoney = ast.currentmoney - (cap * price / self.leverage)
            ast.stocks.append(Stock(datetime, price, cap, 'S', writer_report))

csvfile_r = 'USDJPY_Candlestick_1_h_2017.csv'
csvfile_w = 'Trade_Rsult.csv'
csvfile_report_w = 'Trade_Report.csv'
start_money = 3000000
FR = open(csvfile_r, "r", encoding="utf-8")
FW = open(csvfile_w, "w")
FWR = open(csvfile_report_w, "w")
reader = csv.reader(FR)
writer = csv.writer(FW, lineterminator='\n')
writer_report = csv.writer(FWR, lineterminator='\n')

# header行読み飛ばし
next(reader)
HEADER = ["Date", "Cash", "Stock", "Total", "Profit&Loss"]
writer.writerow(HEADER)
asset = Asset(start_money)
for line in reader:
    lastDatetime = line[0]
    lastPrice = float(line[4])
    if line[0][11:19] == '07:00:00':
        stockvalue = 0
        for stock in asset.stocks:
            stockvalue = round(stock.amount / stock.leverage * float(line[4]), 0)
        totalvalue = round(asset.currentmoney + stockvalue, 0)
        profit = totalvalue - start_money
        report = []
        report.append(line[0])
        report.append(str(asset.currentmoney))
        report.append(str(stockvalue))
        report.append(str(totalvalue))
        report.append(str(profit))
        writer.writerow(report)
    trade_flg = False
    if line[7] != '0':
        if line[7] == '1':
            existFlg = False
            for stock in asset.stocks:
                if stock.position == 'S':
                    stock.settlement(asset, line[0], float(line[4]), writer_report)
                elif stock.position == 'B':
                    existFlg = True
            if existFlg != True:
                Trade().buy(asset, line[0], float(line[4]))
        elif line[7] == '2':
            Trade().buyAll(asset, line[0], float(line[4]))
        elif line[7] == '3':
            existFlg = False
            for stock in asset.stocks:
                if stock.position == 'B':
                    stock.settlement(asset, line[0], float(line[4]), writer_report)
                elif stock.position == 'S':
                    existFlg = True
            if existFlg != True:
                Trade().sell(asset, line[0], float(line[4]))
        elif line[7] == '4':
            Trade().sellAll(asset, line[0], float(line[4]))
        else:
            pass
    else:
        for stock in asset.stocks:
            stock.settlement(asset, line[0], float(line[4]), writer_report)
for stock in asset.stocks:
    stock.settlement(asset, lastDatetime, lastPrice, writer_report)
FR.close()
FW.close()
FWR.close()
