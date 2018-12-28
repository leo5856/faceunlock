from __future__ import unicode_literals  
import mysql.connector
import numpy as np
import sys

class database_op:
	def __init__(self):
		self.db=mysql.connector.connect(user='root',passwd='123',database='face')
		self.cursor = self.db.cursor()
		print ('Face Database Connected')


	def select(self,name):
		self.cursor.execute('select idx from names where name = %s;',([name]))
		values = self.cursor.fetchall()
		if len(values):
			return values[0][0]
		return -1

	def insert(self,name):
		index=self.select(name)
		if index>=0:
			return index
		else:
			self.cursor.execute('update Nums set num=num+1;')
			self.cursor.execute('select * from Nums;')
			values = self.cursor.fetchall()[0][0]
			self.cursor.execute('insert into names values (%s,%s)',([name,values]))
			self.db.commit()
			return values

	def select_all(self):
		self.cursor.execute('select * from names;')
		values = self.cursor.fetchall()
		values.sort(key=lambda a:a[1])
		names=[i[0] for i in values]
		return names

	def __del__(self):
		self.db.close()

