import xlrd

data=xlrd.open_workbook("yzy.xlsx")
#如果是中文名需要先转码
#filename = filename.decode('utf-8')   #解码成utf-8

############获取表#############

table1=data.sheets()[0]    #通过索引顺序获取页表
table2=data.sheet_by_index(1)   #通过索引顺序获取页表
table3=data.sheet_by_name("三班")   #通过页表名获取页表

names=data.sheet_names()   #返回页表名字的列表
print(names)

print(data.sheet_loaded("一班"))   #检查sheet是否导入完毕

############行操作#############

nrows=table1.nrows  #返回sheet1的行数
print(nrows)

print(table1.row(2))   #返回sheet1第3行数据类型和数值的列表

print(table2.row_types(2,start_colx=0, end_colx=None))   #返回该行所有单元格数据类型组成的列表

print(table3.row_values(1,start_colx=0, end_colx=None))   #返回该行所有单元格数值的列表

print(table1.row_len(0)) #返回sheet1第1行的有效单元格长度

############列操作############

print(table1.ncols) #返回该列的有效列数

print(table2.col(1, start_rowx=0, end_rowx=None))   #返回sheet2第2列的数据类型和值的列表

print(table3.col_types(0, start_rowx=0, end_rowx=None))    #返回该列所有单元格数据类型组成的列表

print(table3.col_values(1, start_rowx=0, end_rowx=None))   #返回由该列中所有单元格数值的列表

############单元格操作############

print(table1.cell(1,2))   #返回sheet1第二行第三列的数据类型和值

print(table2.cell_type(1,2))   #返回单元格的数据组成

print(table1.cell_value(3,1))   #返回单元格的值
